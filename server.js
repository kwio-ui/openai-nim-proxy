// server.js - OpenAI to NVIDIA NIM API Proxy
const express = require('express');
const cors = require('cors');
const axios = require('axios');

const THINK_OPEN  = '<think>\n';
const THINK_CLOSE = '\n</think>\n\n';

const app = express();
const PORT = process.env.PORT || 3000;

app.use(cors());
app.use(express.json({ limit: '50mb' }));
app.use(express.urlencoded({ limit: '50mb', extended: true }));

const NIM_API_BASE = process.env.NIM_API_BASE || 'https://integrate.api.nvidia.com/v1';
const NIM_API_KEY  = process.env.NIM_API_KEY;

const SHOW_REASONING = true;

// GLM-5.2 and these models only accept a strict whitelist of fields.
const STRICT_MODELS = [
  'z-ai/glm-5.2',
  'z-ai/glm-5.1',
  'z-ai/glm4.7',
];

// Exact whitelist of fields GLM-5.2 accepts
const STRICT_ALLOWED_FIELDS = ['model', 'messages', 'temperature', 'top_p', 'max_tokens', 'stream'];

// Models that need chat_template_kwargs to activate thinking
const THINKING_REQUIRED_MODELS = [
  'z-ai/glm-5.2',
  'z-ai/glm-5.1',
  'z-ai/glm4.7',
  'deepseek-ai/deepseek-v3.2',
  'deepseek-ai/deepseek-r1',
  'deepseek-ai/deepseek-r1-distill-qwen-32b',
  'deepseek-ai/deepseek-v4-pro',
  'nvidia/nemotron-3-ultra-550b-a55b',
  'nvidia/nemotron-3-nano-omni-30b-a3b-reasoning',
];

// Models that return thinking inline as <think>...</think> in content
// instead of a separate reasoning_content field
const INLINE_THINKING_MODELS = [
  'moonshotai/kimi-k2.6',
  'moonshotai/kimi-k2-thinking',
  'z-ai/glm-5.3',
  'z-ai/glm-5.1',
  'z-ai/glm4.7',
];

// All known valid NIM model names — bypass live test and fallback for these
const ALL_KNOWN_NIM_MODELS = [
  'z-ai/glm-5.2',
  'z-ai/glm-5.1',
  'z-ai/glm4.7',
  'moonshotai/kimi-k2.6',
  'moonshotai/kimi-k2-thinking',
  'deepseek-ai/deepseek-v3.2',
  'deepseek-ai/deepseek-r1',
  'deepseek-ai/deepseek-r1-distill-qwen-32b',
  'deepseek-ai/deepseek-v4-pro',
  'deepseek-ai/deepseek-v4-flash',
  'nvidia/nemotron-3-ultra-550b-a55b',
  'nvidia/nemotron-3-nano-omni-30b-a3b-reasoning',
  'qwen/qwen3.5-397b-a17b',
  'google/diffusiongemma-26b-a4b-it',
  'minimaxai/minimax-m3',
  'mistralai/mistral-large-3-675b-instruct-2512',
  'meta/llama-3.1-405b-instruct',
  'meta/llama-3.1-70b-instruct',
  'meta/llama-3.1-8b-instruct',
];

const MODEL_MAPPING = {
  'gpt-4-turbo':    'moonshotai/kimi-k2.6',
  'gpt-4':          'nvidia/nemotron-3-nano-omni-30b-a3b-reasoning',
  'claude-3-opus':  'deepseek-ai/deepseek-v4-flash',
  'gpt-4o':         'moonshotai/kimi-k2-thinking',
  'gemini-pro':     'nvidia/nemotron-3-ultra-550b-a55b',
  'gemini-1.5-pro': 'qwen/qwen3.5-397b-a17b',
  'gemini-1.6-pro': 'google/diffusiongemma-26b-a4b-it',
  'gemini-1.7-pro': 'minimaxai/minimax-m3',
  'gemini-1.8-pro': 'deepseek-ai/deepseek-v4-pro',
  'gemini-1.9-pro': 'z-ai/glm-5.2',
  'gpt-4o-mini':    'meta/llama-3.1-70b-instruct',
  'claude-3-sonnet':'meta/llama-3.1-70b-instruct',
  'gpt-3.5-turbo':  'meta/llama-3.1-8b-instruct',
  'gemini-2.6-pro': 'mistralai/mistral-large-3-675b-instruct-2512',
  'o1-mini':        'deepseek-ai/deepseek-r1-distill-qwen-32b',
};

async function resolveModel(model) {
  // If it's already a known NIM model, return as-is — no live test, no fallback
  if (ALL_KNOWN_NIM_MODELS.includes(model)) return model;

  // Check alias mapping
  if (MODEL_MAPPING[model]) return MODEL_MAPPING[model];

  // Try live test for unknown models
  try {
    const test = await axios.post(
      `${NIM_API_BASE}/chat/completions`,
      { model, messages: [{ role: 'user', content: 'test' }], max_tokens: 1 },
      {
        headers: { Authorization: `Bearer ${NIM_API_KEY}`, 'Content-Type': 'application/json' },
        validateStatus: s => s < 500,
      }
    );
    if (test.status >= 200 && test.status < 300) return model;
  } catch (_) {}

  // Last resort keyword fallback
  const lower = model.toLowerCase();
  if (lower.includes('gpt-4') || lower.includes('claude-opus') || lower.includes('405b'))
    return 'meta/llama-3.1-405b-instruct';
  if (lower.includes('claude') || lower.includes('gemini') || lower.includes('70b'))
    return 'meta/llama-3.1-70b-instruct';
  return 'meta/llama-3.1-8b-instruct';
}

// Sanitize messages to avoid 400 errors on strict models:
// - Remove invalid roles
// - Remove empty content
// - Ensure system message is first only
// - Ensure strict user/assistant alternation
function sanitizeMessages(messages) {
  const VALID_ROLES = ['system', 'user', 'assistant'];

  // Filter invalid roles and empty content
  let filtered = messages.filter(m => {
    if (!VALID_ROLES.includes(m.role)) return false;
    if (!m.content || m.content.toString().trim() === '') return false;
    return true;
  });

  // Pull out system message (first one only)
  const systemMsg = filtered.find(m => m.role === 'system');
  const rest = filtered.filter(m => m.role !== 'system');

  // Enforce strict user/assistant alternation
  const alternated = [];
  let lastRole = null;
  for (const msg of rest) {
    if (msg.role === lastRole) {
      // Merge consecutive same-role messages
      if (alternated.length > 0) {
        alternated[alternated.length - 1].content += '\n' + msg.content;
      }
    } else {
      alternated.push({ role: msg.role, content: msg.content });
      lastRole = msg.role;
    }
  }

  // Must start with user
  if (alternated.length > 0 && alternated[0].role === 'assistant') {
    alternated.shift();
  }

  return systemMsg ? [systemMsg, ...alternated] : alternated;
}

function buildNimRequest(nimModel, fullBody, useStream) {
  const isStrict   = STRICT_MODELS.includes(nimModel);
  const needsThink = THINKING_REQUIRED_MODELS.includes(nimModel);

  const sanitizedMessages = sanitizeMessages(fullBody.messages || []);

  let body;

  if (isStrict) {
    // WHITELIST ONLY — strip every field GLM rejects
    body = {};
    for (const field of STRICT_ALLOWED_FIELDS) {
      if (fullBody[field] !== undefined) body[field] = fullBody[field];
    }
    body.model       = nimModel;
    body.stream      = useStream;
    body.messages    = sanitizedMessages;
    body.max_tokens  = fullBody.max_tokens || 16384;
    body.temperature = fullBody.temperature || 1;
    body.top_p       = 1;

    // Enable thinking for strict models that support it
    if (needsThink) {
      body.chat_template_kwargs = { enable_thinking: true };
    }
  } else {
    body = {
      model:             nimModel,
      messages:          sanitizedMessages,
      temperature:       fullBody.temperature || 0.7,
      max_tokens:        fullBody.max_tokens  || 16384,
      top_p:             fullBody.top_p       || 0.95,
      stream:            useStream,
      frequency_penalty: fullBody.frequency_penalty || 0,
      presence_penalty:  fullBody.presence_penalty  || 0,
    };

    if (needsThink) {
      body.chat_template_kwargs = { enable_thinking: true, thinking: true };
      body.reasoning_budget = 16384;
    }
  }

  return body;
}

function makeContentChunk(content, baseData) {
  return {
    id:      (baseData && baseData.id)      ? baseData.id      : ('chatcmpl-' + Date.now()),
    object:  (baseData && baseData.object)  ? baseData.object  : 'chat.completion.chunk',
    created: (baseData && baseData.created) ? baseData.created : Math.floor(Date.now() / 1000),
    model:   (baseData && baseData.model)   ? baseData.model   : '',
    choices: [{ index: 0, delta: { content: content }, finish_reason: null }],
  };
}

app.get('/health', function (req, res) {
  res.json({
    status: 'ok',
    service: 'OpenAI to NVIDIA NIM Proxy',
    reasoning_display: SHOW_REASONING,
    strict_models: STRICT_MODELS,
    inline_thinking_models: INLINE_THINKING_MODELS,
    thinking_required_models: THINKING_REQUIRED_MODELS,
  });
});

app.get('/v1/models', function (req, res) {
  res.json({
    object: 'list',
    data: Object.keys(MODEL_MAPPING).map(function (id) {
      return { id: id, object: 'model', created: Date.now(), owned_by: 'nvidia-nim-proxy' };
    }),
  });
});

app.post('/v1/chat/completions', async function (req, res) {
  try {
    const { model, stream } = req.body;
    const useStream = stream !== false;

    const nimModel   = await resolveModel(model);
    const nimRequest = buildNimRequest(nimModel, req.body, useStream);

    const isInlineThinking = INLINE_THINKING_MODELS.includes(nimModel);

    console.log(`[proxy] ${model} → ${nimModel} | stream=${useStream} | strict=${STRICT_MODELS.includes(nimModel)} | inlineThink=${isInlineThinking}`);
    console.log(`[proxy] sending fields: ${Object.keys(nimRequest).join(', ')}`);

    const nimResponse = await axios.post(
      `${NIM_API_BASE}/chat/completions`,
      nimRequest,
      {
        headers: {
          Authorization: `Bearer ${NIM_API_KEY}`,
          'Content-Type': 'application/json',
        },
        responseType: useStream ? 'stream' : 'json',
      }
    );

    // ── Streaming ─────────────────────────────────────────────────────────────
    if (useStream) {
      res.setHeader('Content-Type', 'text/event-stream');
      res.setHeader('Cache-Control', 'no-cache');
      res.setHeader('Connection', 'keep-alive');

      // ── All state is declared per-request inside here (no leaking between requests) ──
      var buffer         = '';
      var reasoningOpen  = false;
      var thinkingClosed = false;
      var lastBaseData   = null;

      // State for inline <think>...</think> models (Kimi, GLM)
      var inlineThinkBuffer = '';
      var inlineThinkOpen   = false;
      var inlineThinkDone   = false;

      function emitSynthetic(content) {
        if (!content) return;
        res.write('data: ' + JSON.stringify(makeContentChunk(content, lastBaseData)) + '\n\n');
      }

      function closeThinkBlock() {
        if (reasoningOpen && !thinkingClosed) {
          emitSynthetic(THINK_CLOSE);
          thinkingClosed = true;
          reasoningOpen  = false;
        }
      }

      // Handle content for models that return <think> tags inline in content
      function handleInlineThinkContent(content) {
        if (!content) return;

        inlineThinkBuffer += content;

        if (!inlineThinkDone) {
          if (!inlineThinkOpen) {
            // Check if think tag is starting
            if (inlineThinkBuffer.includes('<think>')) {
              inlineThinkOpen = true;
              const parts = inlineThinkBuffer.split('<think>');
              // Anything before <think> is normal content
              if (parts[0]) emitSynthetic(parts[0]);
              // Emit THINK_OPEN and content after <think>
              const afterOpen = parts.slice(1).join('<think>');
              emitSynthetic(THINK_OPEN + afterOpen);
              inlineThinkBuffer = afterOpen;
            } else if (!('<think>'.startsWith(inlineThinkBuffer.slice(-7)))) {
              // No think tag coming — flush as normal content
              inlineThinkDone = true;
              emitSynthetic(inlineThinkBuffer);
              inlineThinkBuffer = '';
            }
            // else: partial '<think>' at end of buffer, keep buffering
          } else {
            // Inside <think>, watching for </think>
            if (inlineThinkBuffer.includes('</think>')) {
              inlineThinkDone = true;
              const parts = inlineThinkBuffer.split('</think>');
              // Emit closing of think block
              emitSynthetic(parts[0] + THINK_CLOSE);
              thinkingClosed = true;
              reasoningOpen  = false;
              inlineThinkBuffer = '';
              // Emit everything after </think> as normal content
              const afterClose = parts.slice(1).join('</think>');
              if (afterClose) emitSynthetic(afterClose);
            } else {
              // Still inside think, stream it live
              emitSynthetic(content);
            }
          }
        } else {
          // Past think block — normal content passthrough
          emitSynthetic(content);
          inlineThinkBuffer = '';
        }
      }

      nimResponse.data.on('data', function (chunk) {
        buffer += chunk.toString();
        var lines = buffer.split('\n');
        buffer = lines.pop() || '';

        for (var i = 0; i < lines.length; i++) {
          var line = lines[i];
          if (!line.startsWith('data: ')) continue;

          if (line.includes('[DONE]')) {
            closeThinkBlock();
            res.write('data: [DONE]\n\n');
            continue;
          }

          try {
            var data = JSON.parse(line.slice(6));
            lastBaseData = data;

            var delta = (data.choices && data.choices[0] && data.choices[0].delta)
              ? data.choices[0].delta
              : null;

            if (!delta) {
              res.write('data: ' + JSON.stringify(data) + '\n\n');
              continue;
            }

            var reasoning = delta.reasoning_content || '';
            var content   = delta.content           || '';
            delete delta.reasoning_content;

            if (SHOW_REASONING) {

              // ── Inline thinking models (Kimi, GLM): <think> in content field ──
              if (isInlineThinking && content) {
                handleInlineThinkContent(content);
                continue;
              }

              // ── Models with separate reasoning_content field (DeepSeek etc) ──
              var combined = '';

              if (reasoning) {
                if (!reasoningOpen) {
                  combined += THINK_OPEN + reasoning;
                  reasoningOpen = true;
                } else {
                  combined += reasoning;
                }
              }

              if (content) {
                if (reasoningOpen && !thinkingClosed) {
                  combined += THINK_CLOSE + content;
                  thinkingClosed = true;
                  reasoningOpen  = false;
                } else {
                  combined += content;
                }
              }

              if (!combined) continue;
              delta.content = combined;

            } else {
              // SHOW_REASONING = false: skip reasoning, only pass content
              if (!content) continue;
              delta.content = content;
            }

            res.write('data: ' + JSON.stringify(data) + '\n\n');

          } catch (parseErr) {
            res.write(line + '\n');
          }
        }
      });

      nimResponse.data.on('end', function () {
        closeThinkBlock();
        res.end();
      });

      nimResponse.data.on('error', function (err) {
        console.error('Stream error:', err);
        closeThinkBlock();
        res.end();
      });

    // ── Non-streaming ─────────────────────────────────────────────────────────
    } else {
      var choices = nimResponse.data.choices.map(function (choice) {
        var rawContent = (choice.message && choice.message.content)
          ? choice.message.content : '';

        var finalContent = rawContent;

        if (SHOW_REASONING) {
          // Models with separate reasoning_content field
          if (choice.message && choice.message.reasoning_content) {
            finalContent = THINK_OPEN + choice.message.reasoning_content + THINK_CLOSE + rawContent;
          }
          // Inline thinking models: reformat <think>...</think> to proxy format
          else if (INLINE_THINKING_MODELS.includes(nimModel) && rawContent.includes('<think>')) {
            finalContent = rawContent
              .replace('<think>', THINK_OPEN)
              .replace('</think>', THINK_CLOSE);
          }
        } else {
          // Strip inline think tags entirely when SHOW_REASONING = false
          finalContent = rawContent.replace(/<think>[\s\S]*?<\/think>\s*/g, '');
        }

        return {
          index:         choice.index,
          message:       { role: choice.message.role, content: finalContent },
          finish_reason: choice.finish_reason,
        };
      });

      res.json({
        id:      'chatcmpl-' + Date.now(),
        object:  'chat.completion',
        created: Math.floor(Date.now() / 1000),
        model:   model,
        choices: choices,
        usage:   nimResponse.data.usage || { prompt_tokens: 0, completion_tokens: 0, total_tokens: 0 },
      });
    }

  } catch (error) {
    console.error('Proxy error:', error.message);

    if (error.response) {
      console.error('NIM status:', error.response.status);
      if (error.response.data) {
        if (typeof error.response.data.on === 'function') {
          let raw = '';
          error.response.data.on('data', c => { raw += c.toString(); });
          error.response.data.on('end',  () => {
            console.error('NIM error body:', raw);
            console.error('NIM request was:', JSON.stringify(error.config && error.config.data, null, 2));
          });
        } else {
          console.error('NIM error body:', JSON.stringify(error.response.data));
          console.error('NIM request was:', JSON.stringify(error.config && error.config.data, null, 2));
        }
      }
    }

    res.status((error.response && error.response.status) || 500).json({
      error: {
        message: error.message || 'Internal server error',
        type:    'invalid_request_error',
        code:    (error.response && error.response.status) || 500,
      },
    });
  }
});

app.all('*', function (req, res) {
  res.status(404).json({
    error: { message: 'Endpoint ' + req.path + ' not found', type: 'invalid_request_error', code: 404 },
  });
});

app.listen(PORT, function () {
  console.log('OpenAI to NVIDIA NIM Proxy running on port ' + PORT);
  console.log('Health check: http://localhost:' + PORT + '/health');
  console.log('Reasoning display: ' + (SHOW_REASONING ? 'ENABLED' : 'DISABLED'));
});
