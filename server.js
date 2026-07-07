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
// Any extra field (frequency_penalty, presence_penalty, logit_bias, n,
// user, stop, best_of, etc.) sent by JanitorAI causes 410.
const STRICT_MODELS = [
  'z-ai/glm-5.2',
  'z-ai/glm-5.1',
  'z-ai/glm4.7',
];

// Exact whitelist of fields GLM-5.2 accepts — nothing else allowed
const STRICT_ALLOWED_FIELDS = ['model', 'messages', 'temperature', 'top_p', 'max_tokens', 'seed', 'stream'];

// Models that need chat_template_kwargs to activate thinking
const THINKING_REQUIRED_MODELS = [
  'deepseek-ai/deepseek-v3.2',
  'deepseek-ai/deepseek-r1',
  'deepseek-ai/deepseek-r1-distill-qwen-32b',
  'deepseek-ai/deepseek-v4-pro',
  'nvidia/nemotron-3-ultra-550b-a55b',
  'nvidia/nemotron-3-nano-omni-30b-a3b-reasoning',
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
  if (MODEL_MAPPING[model]) return MODEL_MAPPING[model];

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

  const lower = model.toLowerCase();
  if (lower.includes('gpt-4') || lower.includes('claude-opus') || lower.includes('405b'))
    return 'meta/llama-3.1-405b-instruct';
  if (lower.includes('claude') || lower.includes('gemini') || lower.includes('70b'))
    return 'meta/llama-3.1-70b-instruct';
  return 'meta/llama-3.1-8b-instruct';
}

function buildNimRequest(nimModel, fullBody, useStream) {
  const isStrict   = STRICT_MODELS.includes(nimModel);
  const needsThink = THINKING_REQUIRED_MODELS.includes(nimModel);

  let body;

  if (isStrict) {
    // WHITELIST ONLY — strip every field JanitorAI sends that GLM-5.2 rejects
    body = {};
    for (const field of STRICT_ALLOWED_FIELDS) {
      if (fullBody[field] !== undefined) body[field] = fullBody[field];
    }
    // Always override model with the resolved NIM model
    body.model      = nimModel;
    body.stream     = useStream;
    body.max_tokens = fullBody.max_tokens || 16384;
    body.temperature = fullBody.temperature || 1;
    body.top_p      = 1;
  } else {
    // For non-strict models pass through common fields
    body = {
      model:             nimModel,
      messages:          fullBody.messages,
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

    // Log exactly what we're sending so you can debug future issues
    console.log(`[proxy] ${model} → ${nimModel} | stream=${useStream} | strict=${STRICT_MODELS.includes(nimModel)}`);
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

      var buffer         = '';
      var reasoningOpen  = false;
      var thinkingClosed = false;
      var lastBaseData   = null;

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
        var finalContent = (choice.message && choice.message.content)
          ? choice.message.content : '';

        if (SHOW_REASONING && choice.message && choice.message.reasoning_content) {
          finalContent = THINK_OPEN + choice.message.reasoning_content + THINK_CLOSE + finalContent;
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
          error.response.data.on('end',  () => { console.error('NIM error body:', raw); });
        } else {
          console.error('NIM error body:', JSON.stringify(error.response.data));
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
