// server.js - OpenAI to NVIDIA NIM API Proxy
const express = require('express');
const cors = require('cors');
const axios = require('axios');

const app = express();
const PORT = process.env.PORT || 3000;

// Middleware
app.use(cors());
// Add this line AFTER: const app = express();
app.use(express.json({ limit: '50mb' }));
app.use(express.urlencoded({ limit: '50mb', extended: true }));
// NVIDIA NIM API configuration
const NIM_API_BASE = process.env.NIM_API_BASE || 'https://integrate.api.nvidia.com/v1';
const NIM_API_KEY = process.env.NIM_API_KEY;

// 🔥 REASONING DISPLAY TOGGLE - Shows/hides reasoning in output
const SHOW_REASONING = true; // Set to true to show reasoning with <think> tags

// 🔥 THINKING MODE TOGGLE - Auto-enabled for models that need it
// Models that REQUIRE thinking parameter (Endpoint Only models)
const THINKING_REQUIRED_MODELS = [
  'deepseek-ai/deepseek-v3.2',
  'deepseek-ai/deepseek-r1',
  'z-ai/glm-5.1',
  'deepseek-ai/deepseek-r1-distill-qwen-32b',
  'google/gemma-4-31b-it', 
  'deepseek-ai/deepseek-v4-pro'
];

// Model mapping (adjust based on available NIM models)
// 📝 Memory | 🎭 Character Consistency | ⚡ Speed | 🎨 Creativity
const MODEL_MAPPING = {
  // 🏆 BEST FOR JANITOR AI ROLEPLAY - Long memory + Character consistency
  'gpt-4-turbo': 'moonshotai/kimi-k2.6',                     // 📝 1T MoE, video/image understanding, NEWEST!
  'gpt-4': 'nvidia/llama-3.1-nemotron-ultra-253b-v1',        // 🎭 Best character consistency
  'claude-3-opus': 'nvidia/llama-3.1-nemotron-ultra-253b-v1', // 🎭 Same reliability
  
  // High quality with good memory
  'gpt-4o': 'moonshotai/kimi-k2-thinking',                   // 📝 256k context, native reasoning
  'gemini-pro': 'nvidia/nemotron-3-ultra-550b-a55b',
  'gemini-1.5-pro': 'z-ai/glm4.7',// 📝 Very capable, good memory
  'gemini-1.6-pro': 'stepfun-ai/step-3.5-flash',
  'gemini-1.7-pro': 'google/gemma-4-31b-it', 
  'gemini-1.8-pro': 'deepseek-ai/deepseek-v4-pro',
  'gemini-1.9-pro': 'z-ai/glm-5.1', 
  

  // Fast but still good quality  
  'gpt-4o-mini': 'meta/llama-3.1-70b-instruct',              // ⚡ Fast, decent memory
  'claude-3-sonnet': 'meta/llama-3.1-70b-instruct',          // ⚡ Alternative fast option
  
  // Ultra-fast for testing
  'gpt-3.5-turbo': 'meta/llama-3.1-8b-instruct',
  'gemini-2.6-pro': 'mistralai/mistral-large-3-675b-instruct-2512',// ⚡⚡⚡ Fastest
  
  // Thinking models (slow but very smart)
  'o1': 'deepseek-ai/deepseek-v3.2', // 🧠 685B reasoning model
  'o1-mini': 'deepseek-ai/deepseek-r1-distill-qwen-32b'     // 🧠 Faster thinking
};

// Health check endpoint
app.get('/health', (req, res) => {
  res.json({ 
    status: 'ok', 
    service: 'OpenAI to NVIDIA NIM Proxy', 
    reasoning_display: SHOW_REASONING,
    thinking_models: THINKING_REQUIRED_MODELS
  });
});

// List models endpoint (OpenAI compatible)
app.get('/v1/models', (req, res) => {
  const models = Object.keys(MODEL_MAPPING).map(model => ({
    id: model,
    object: 'model',
    created: Date.now(),
    owned_by: 'nvidia-nim-proxy'
  }));
  
  res.json({
    object: 'list',
    data: models
  });
});

// Chat completions endpoint (main proxy)
app.post('/v1/chat/completions', async (req, res) => {
  try {
    const { model, messages, temperature, max_tokens, stream } = req.body;
    
    // Smart model selection with fallback
    let nimModel = MODEL_MAPPING[model];
    if (!nimModel) {
      try {
        await axios.post(`${NIM_API_BASE}/chat/completions`, {
          model: model,
          messages: [{ role: 'user', content: 'test' }],
          max_tokens: 1
        }, {
          headers: { 'Authorization': `Bearer ${NIM_API_KEY}`, 'Content-Type': 'application/json' },
          validateStatus: (status) => status < 500
        }).then(res => {
          if (res.status >= 200 && res.status < 300) {
            nimModel = model;
          }
        });
      } catch (e) {}
      
      if (!nimModel) {
        const modelLower = model.toLowerCase();
        if (modelLower.includes('gpt-4') || modelLower.includes('claude-opus') || modelLower.includes('405b')) {
          nimModel = 'meta/llama-3.1-405b-instruct';
        } else if (modelLower.includes('claude') || modelLower.includes('gemini') || modelLower.includes('70b')) {
          nimModel = 'meta/llama-3.1-70b-instruct';
        } else {
          nimModel = 'meta/llama-3.1-8b-instruct';
        }
      }
    }
    
    // Transform OpenAI request to NIM format
    // Auto-enable thinking for "Endpoint Only" models
    const requiresThinking = THINKING_REQUIRED_MODELS.includes(nimModel);
    
    const nimRequest = {
      model: nimModel,
      messages: messages,
      temperature: temperature || 0.7,
      max_tokens: max_tokens || 20000,
      top_p: 0.95,
      frequency_penalty: 0.0,
      presence_penalty: 0.0,
      stop: null,
      chat_template_kwargs: {thinking:true, clear_thinking:true, do_sample:true, enable_thinking:true, clear_thinking:true},
      stream: stream || true
    };
    
    // Make request to NVIDIA NIM API
    const response = await axios.post(`${NIM_API_BASE}/chat/completions`, nimRequest, {
      headers: {
        'Authorization': `Bearer ${NIM_API_KEY}`,
        'Content-Type': 'application/json'
      },
      responseType: stream ? 'stream' : 'json'
    });
    
    if (stream) {
      // Handle streaming response with reasoning
      res.setHeader('Content-Type', 'text/event-stream');
      res.setHeader('Cache-Control', 'no-cache');
      res.setHeader('Connection', 'keep-alive');
      
      let buffer = '';
      let reasoningStarted = false;
      
      response.data.on('data', (chunk) => {
        buffer += chunk.toString();
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';
        
        lines.forEach(line => {
          if (line.startsWith('data: ')) {
            if (line.includes('[DONE]')) {
              res.write(line + '\n');
              return;
            }
            
            try {
              const data = JSON.parse(line.slice(6));
              if (data.choices?.[0]?.delta) {
                const reasoning = data.choices[0].delta.reasoning_content;
                const content = data.choices[0].delta.content;
                
                if (SHOW_REASONING) {
                  let combinedContent = '';
                  
                  if (reasoning && !reasoningStarted) {
                    combinedContent = '<think>\n' + reasoning;
                    reasoningStarted = true;
                  } else if (reasoning) {
                    combinedContent = reasoning;
                  }
                  
                  if (content && reasoningStarted) {
                    combinedContent += '</think>\n\n' + content;
                    reasoningStarted = false;
                  } else if (content) {
                    combinedContent += content;
                  }
                  
                  if (combinedContent) {
                    data.choices[0].delta.content = combinedContent;
                    delete data.choices[0].delta.reasoning_content;
                  }
                } else {
                  if (content) {
                    data.choices[0].delta.content = content;
                  } else {
                    data.choices[0].delta.content = '';
                  }
                  delete data.choices[0].delta.reasoning_content;
                }
              }
              res.write(`data: ${JSON.stringify(data)}\n\n`);
            } catch (e) {
              res.write(line + '\n');
            }
          }
        });
      });
      
      response.data.on('end', () => res.end());
      response.data.on('error', (err) => {
        console.error('Stream error:', err);
        res.end();
      });
    } else {
      // Transform NIM response to OpenAI format with reasoning
      const openaiResponse = {
        id: `chatcmpl-${Date.now()}`,
        object: 'chat.completion',
        created: Math.floor(Date.now() / 1000),
        model: model,
        choices: response.data.choices.map(choice => {
          let fullContent = choice.message?.content || '';
          
          if (SHOW_REASONING && choice.message?.reasoning_content) {
            fullContent = '<think>\n' + choice.message.reasoning_content + '\n</think>\n\n' + fullContent;
          }
          
          return {
            index: choice.index,
            message: {
              role: choice.message.role,
              content: fullContent
            },
            finish_reason: choice.finish_reason
          };
        }),
        usage: response.data.usage || {
          prompt_tokens: 0,
          completion_tokens: 0,
          total_tokens: 0
        }
      };
      
      res.json(openaiResponse);
    }
    
  } catch (error) {
    console.error('Proxy error:', error.message);
    
    res.status(error.response?.status || 500).json({
      error: {
        message: error.message || 'Internal server error',
        type: 'invalid_request_error',
        code: error.response?.status || 500
      }
    });
  }
});

// Catch-all for unsupported endpoints
app.all('*', (req, res) => {
  res.status(404).json({
    error: {
      message: `Endpoint ${req.path} not found`,
      type: 'invalid_request_error',
      code: 404
    }
  });
});

app.listen(PORT, () => {
  console.log(`OpenAI to NVIDIA NIM Proxy running on port ${PORT}`);
  console.log(`Health check: http://localhost:${PORT}/health`);
  console.log(`Reasoning display: ${SHOW_REASONING ? 'ENABLED' : 'DISABLED'}`);
  console.log(`Thinking models: ${THINKING_REQUIRED_MODELS.join(', ')}`);
});  'gpt-4': 'nvidia/llama-3.1-nemotron-ultra-253b-v1',        // 🎭 Best character consistency
  'claude-3-opus': 'nvidia/llama-3.1-nemotron-ultra-253b-v1', // 🎭 Same reliability
  
  // High quality with good memory
  'gpt-4o': 'moonshotai/kimi-k2-thinking',                   // 📝 256k context, native reasoning
  'gemini-pro': 'nvidia/nemotron-3-ultra-550b-a55b',
  'gemini-1.5-pro': 'z-ai/glm4.7',// 📝 Very capable, good memory
  'gemini-1.6-pro': 'stepfun-ai/step-3.5-flash',
  'gemini-1.7-pro': 'google/gemma-4-31b-it', 
  'gemini-1.8-pro': 'deepseek-ai/deepseek-v4-pro',
  'gemini-1.9-pro': 'z-ai/glm-5.1', 
  

  // Fast but still good quality  
  'gpt-4o-mini': 'meta/llama-3.1-70b-instruct',              // ⚡ Fast, decent memory
  'claude-3-sonnet': 'meta/llama-3.1-70b-instruct',          // ⚡ Alternative fast option
  
  // Ultra-fast for testing
  'gpt-3.5-turbo': 'meta/llama-3.1-8b-instruct',
  'gemini-2.6-pro': 'mistralai/mistral-large-3-675b-instruct-2512',// ⚡⚡⚡ Fastest
  
  // Thinking models (slow but very smart)
  'o1': 'deepseek-ai/deepseek-v3.2', // 🧠 685B reasoning model
  'o1-mini': 'deepseek-ai/deepseek-r1-distill-qwen-32b'     // 🧠 Faster thinking
};

// Health check endpoint
app.get('/health', (req, res) => {
  res.json({ 
    status: 'ok', 
    service: 'OpenAI to NVIDIA NIM Proxy', 
    reasoning_display: SHOW_REASONING,
    thinking_models: THINKING_REQUIRED_MODELS
  });
});

// List models endpoint (OpenAI compatible)
app.get('/v1/models', (req, res) => {
  const models = Object.keys(MODEL_MAPPING).map(model => ({
    id: model,
    object: 'model',
    created: Date.now(),
    owned_by: 'nvidia-nim-proxy'
  }));
  
  res.json({
    object: 'list',
    data: models
  });
});

// Chat completions endpoint (main proxy)
app.post('/v1/chat/completions', async (req, res) => {
  try {
    const { model, messages, temperature, max_tokens, stream } = req.body;
    
    // Smart model selection with fallback
    let nimModel = MODEL_MAPPING[model];
    if (!nimModel) {
      try {
        await axios.post(`${NIM_API_BASE}/chat/completions`, {
          model: model,
          messages: [{ role: 'user', content: 'test' }],
          max_tokens: 1
        }, {
          headers: { 'Authorization': `Bearer ${NIM_API_KEY}`, 'Content-Type': 'application/json' },
          validateStatus: (status) => status < 500
        }).then(res => {
          if (res.status >= 200 && res.status < 300) {
            nimModel = model;
          }
        });
      } catch (e) {}
      
      if (!nimModel) {
        const modelLower = model.toLowerCase();
        if (modelLower.includes('gpt-4') || modelLower.includes('claude-opus') || modelLower.includes('405b')) {
          nimModel = 'meta/llama-3.1-405b-instruct';
        } else if (modelLower.includes('claude') || modelLower.includes('gemini') || modelLower.includes('70b')) {
          nimModel = 'meta/llama-3.1-70b-instruct';
        } else {
          nimModel = 'meta/llama-3.1-8b-instruct';
        }
      }
    }
    
    // Transform OpenAI request to NIM format
    // Auto-enable thinking for "Endpoint Only" models
    const requiresThinking = THINKING_REQUIRED_MODELS.includes(nimModel);
    
    const nimRequest = {
      model: nimModel,
      messages: messages,
      temperature: temperature || 0.7,
      max_tokens: max_tokens || 20000,
      top_p: 0.95,
      frequency_penalty: 0.0,
      presence_penalty: 0.0,
      stop: null,
      chat_template_kwargs: {thinking:true, clear_thinking:true, do_sample:true, enable_thinking:true, clear_thinking:true},
      stream: stream || true
    };
    
    // Make request to NVIDIA NIM API
    const response = await axios.post(`${NIM_API_BASE}/chat/completions`, nimRequest, {
      headers: {
        'Authorization': `Bearer ${NIM_API_KEY}`,
        'Content-Type': 'application/json'
      },
      responseType: stream ? 'stream' : 'json'
    });
    
    if (stream) {
      // Handle streaming response with reasoning
      res.setHeader('Content-Type', 'text/event-stream');
      res.setHeader('Cache-Control', 'no-cache');
      res.setHeader('Connection', 'keep-alive');
      
      let buffer = '';
      let reasoningStarted = false;
      
      response.data.on('data', (chunk) => {
        buffer += chunk.toString();
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';
        
        lines.forEach(line => {
          if (line.startsWith('data: ')) {
            if (line.includes('[DONE]')) {
              res.write(line + '\n');
              return;
            }
            
            try {
              const data = JSON.parse(line.slice(6));
              if (data.choices?.[0]?.delta) {
                const reasoning = data.choices[0].delta.reasoning_content;
                const content = data.choices[0].delta.content;
                
                if (SHOW_REASONING) {
                  let combinedContent = '';
                  
                  if (reasoning && !reasoningStarted) {
                    combinedContent = '<think>\n' + reasoning;
                    reasoningStarted = true;
                  } else if (reasoning) {
                    combinedContent = reasoning;
                  }
                  
                  if (content && reasoningStarted) {
                    combinedContent += '</think>\n\n' + content;
                    reasoningStarted = false;
                  } else if (content) {
                    combinedContent += content;
                  }
                  
                  if (combinedContent) {
                    data.choices[0].delta.content = combinedContent;
                    delete data.choices[0].delta.reasoning_content;
                  }
                } else {
                  if (content) {
                    data.choices[0].delta.content = content;
                  } else {
                    data.choices[0].delta.content = '';
                  }
                  delete data.choices[0].delta.reasoning_content;
                }
              }
              res.write(`data: ${JSON.stringify(data)}\n\n`);
            } catch (e) {
              res.write(line + '\n');
            }
          }
        });
      });
      
      response.data.on('end', () => res.end());
      response.data.on('error', (err) => {
        console.error('Stream error:', err);
        res.end();
      });
    } else {
      // Transform NIM response to OpenAI format with reasoning
      const openaiResponse = {
        id: `chatcmpl-${Date.now()}`,
        object: 'chat.completion',
        created: Math.floor(Date.now() / 1000),
        model: model,
        choices: response.data.choices.map(choice => {
          let fullContent = choice.message?.content || '';
          
          if (SHOW_REASONING && choice.message?.reasoning_content) {
            fullContent = '<think>\n' + choice.message.reasoning_content + '\n</think>\n\n' + fullContent;
          }
          
          return {
            index: choice.index,
            message: {
              role: choice.message.role,
              content: fullContent
            },
            finish_reason: choice.finish_reason
          };
        }),
        usage: response.data.usage || {
          prompt_tokens: 0,
          completion_tokens: 0,
          total_tokens: 0
        }
      };
      
      res.json(openaiResponse);
    }
    
  } catch (error) {
    console.error('Proxy error:', error.message);
    
    res.status(error.response?.status || 500).json({
      error: {
        message: error.message || 'Internal server error',
        type: 'invalid_request_error',
        code: error.response?.status || 500
      }
    });
  }
});

// Catch-all for unsupported endpoints
app.all('*', (req, res) => {
  res.status(404).json({
    error: {
      message: `Endpoint ${req.path} not found`,
      type: 'invalid_request_error',
      code: 404
    }
  });
});

app.listen(PORT, () => {
  console.log(`OpenAI to NVIDIA NIM Proxy running on port ${PORT}`);
  console.log(`Health check: http://localhost:${PORT}/health`);
  console.log(`Reasoning display: ${SHOW_REASONING ? 'ENABLED' : 'DISABLED'}`);
  console.log(`Thinking models: ${THINKING_REQUIRED_MODELS.join(', ')}`);
});  // 🏆 BEST FOR JANITOR AI ROLEPLAY
  'gpt-4-turbo':    'moonshotai/kimi-k2.6',
  'gpt-4':          'nvidia/llama-3.1-nemotron-ultra-253b-v1',
  'claude-3-opus':  'nvidia/llama-3.1-nemotron-ultra-253b-v1',

  // High quality
  'gpt-4o':         'moonshotai/kimi-k2-thinking',
  'gemini-pro':     'nvidia/nemotron-3-ultra-550b-a55b',
  'gemini-1.5-pro': 'z-ai/glm4.7',
  'gemini-1.6-pro': 'stepfun-ai/step-3.5-flash',
  'gemini-1.7-pro': 'google/gemma-4-31b-it',
  'gemini-1.8-pro': 'deepseek-ai/deepseek-v4-pro',
  'gemini-1.9-pro': 'z-ai/glm-5.1',

  // Fast
  'gpt-4o-mini':      'meta/llama-3.1-70b-instruct',
  'claude-3-sonnet':  'meta/llama-3.1-70b-instruct',

  // Ultra-fast / testing
  'gpt-3.5-turbo':  'meta/llama-3.1-8b-instruct',
  'gemini-2.6-pro': 'mistralai/mistral-large-3-675b-instruct-2512',

  // Thinking models
  'o1':      'deepseek-ai/deepseek-v3.2',
  'o1-mini': 'deepseek-ai/deepseek-r1-distill-qwen-32b'
};

// ─── Helpers ──────────────────────────────────────────────────────────────────

/**
 * Resolve the NIM model string from the incoming OpenAI model name.
 * Falls back gracefully if the model is unknown.
 */
async function resolveModel(model) {
  if (MODEL_MAPPING[model]) return MODEL_MAPPING[model];

  // Try the model name directly against NIM
  try {
    const test = await axios.post(
      `${NIM_API_BASE}/chat/completions`,
      { model, messages: [{ role: 'user', content: 'test' }], max_tokens: 1 },
      {
        headers: { Authorization: `Bearer ${NIM_API_KEY}`, 'Content-Type': 'application/json' },
        validateStatus: s => s < 500
      }
    );
    if (test.status >= 200 && test.status < 300) return model;
  } catch (_) {}

  // Heuristic fallback
  const lower = model.toLowerCase();
  if (lower.includes('gpt-4') || lower.includes('claude-opus') || lower.includes('405b'))
    return 'meta/llama-3.1-405b-instruct';
  if (lower.includes('claude') || lower.includes('gemini') || lower.includes('70b'))
    return 'meta/llama-3.1-70b-instruct';
  return 'meta/llama-3.1-8b-instruct';
}

/**
 * Build a minimal SSE data chunk with just a content delta.
 */
function makeContentChunk(content, baseData) {
  return {
    id:      baseData?.id      || `chatcmpl-${Date.now()}`,
    object:  baseData?.object  || 'chat.completion.chunk',
    created: baseData?.created || Math.floor(Date.now() / 1000),
    model:   baseData?.model   || '',
    choices: [{
      index: 0,
      delta: { content },
      finish_reason: null
    }]
  };
}

// ─── Routes ───────────────────────────────────────────────────────────────────

app.get('/health', (req, res) => {
  res.json({
    status:           'ok',
    service:          'OpenAI → NVIDIA NIM Proxy',
    reasoning_display: SHOW_REASONING,
    thinking_models:  THINKING_REQUIRED_MODELS
  });
});

app.get('/v1/models', (req, res) => {
  res.json({
    object: 'list',
    data: Object.keys(MODEL_MAPPING).map(id => ({
      id,
      object:    'model',
      created:   Date.now(),
      owned_by:  'nvidia-nim-proxy'
    }))
  });
});

// ─── Main proxy endpoint ──────────────────────────────────────────────────────
app.post('/v1/chat/completions', async (req, res) => {
  try {
    const { model, messages, temperature, max_tokens, stream } = req.body;
    const useStream = stream !== false; // default true

    const nimModel = await resolveModel(model);

    const nimRequest = {
      model:    nimModel,
      messages: messages,
      temperature:       temperature   || 0.7,
      max_tokens:        max_tokens    || 20000,
      top_p:             0.95,
      frequency_penalty: 0.0,
      presence_penalty:  0.0,
      stop:              null,
      chat_template_kwargs: {
        thinking:        true,
        clear_thinking:  true,
        do_sample:       true,
        enable_thinking: true
      },
      stream: useStream
    };

    const nimResponse = await axios.post(
      `${NIM_API_BASE}/chat/completions`,
      nimRequest,
      {
        headers: {
          Authorization:  `Bearer ${NIM_API_KEY}`,
          'Content-Type': 'application/json'
        },
        responseType: useStream ? 'stream' : 'json'
      }
    );

    // ── Streaming path ────────────────────────────────────────────────────────
    if (useStream) {
      res.setHeader('Content-Type',  'text/event-stream');
      res.setHeader('Cache-Control', 'no-cache');
      res.setHeader('Connection',    'keep-alive');

      let buffer         = '';
      let reasoningOpen  = false; // true while inside a <think> block
      let thinkingClosed = false; // true once </think> has been emitted
      let lastBaseData   = null;  // keep a ref for synthetic chunks

      /**
       * Emit a synthetic SSE chunk containing only the given content string.
       * Used to inject <think>, </think>, or the closing safety-net tag.
       */
      function emitSynthetic(content) {
        if (!content) return;
        const chunk = makeContentChunk(content, lastBaseData);
        res.write(`data: ${JSON.stringify(chunk)}\n\n`);
      }

      /**
       * Close the think block exactly once.
       */
      function closeThinkBlock() {
        if (reasoningOpen && !thinkingClosed) {
          emitSynthetic('</think>\n\n');
          thinkingClosed = true;
          reasoningOpen  = false;
        }
      }

      nimResponse.data.on('data', (chunk) => {
        buffer += chunk.toString();
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';

        for (const line of lines) {
          if (!line.startsWith('data: ')) continue;

          // ── [DONE] sentinel ────────────────────────────────────────────────
          if (line.includes('[DONE]')) {
            closeThinkBlock(); // safety-net: close if still open
            res.write('data: [DONE]\n\n');
            continue;
          }

          try {
            const data = JSON.parse(line.slice(6));
            lastBaseData = data; // cache for synthetic chunk metadata

            const delta     = data.choices?.[0]?.delta;
            const reasoning = delta?.reasoning_content || '';
            const content   = delta?.content           || '';

            // Always remove the raw reasoning_content field from the delta
            // before forwarding — we fold it into content manually.
            if (delta) delete delta.reasoning_content;

            if (SHOW_REASONING) {
              let combined = '';

              // ── Reasoning token arrived ──────────────────────────────────
              if (reasoning) {
                if (!reasoningOpen) {
                  combined      += '<think>\n' + reasoning;
                  reasoningOpen  = true;
                } else {
                  combined += reasoning;
                }
              }

              // ── Content token arrived ────────────────────────────────────
              if (content) {
                if (reasoningOpen && !thinkingClosed) {
                  // Close the think block on the first content token
                  combined      += '</think>\n\n' + content;
                  thinkingClosed = true;
                  reasoningOpen  = false;
                } else {
                  combined += content;
                }
              }

              if (combined) {
                delta.content = combined;
              } else {
                // Nothing to forward this tick — skip the chunk entirely
                // to avoid spamming empty deltas
                continue;
              }

            } else {
              // SHOW_REASONING = false → strip all reasoning, pass content only
              if (!content) continue;
              delta.content = content;
            }

            res.write(`data: ${JSON.stringify(data)}\n\n`);

          } catch (parseErr) {
            // Malformed JSON line — forward as-is
            res.write(line + '\n');
          }
        }
      });

      nimResponse.data.on('end', () => {
        // Final safety-net: if the model finished inside a think block
        // (e.g. content never arrived), close it now.
        closeThinkBlock();
        res.end();
      });

      nimResponse.data.on('error', (err) => {
        console.error('Stream error:', err);
        closeThinkBlock();
        res.end();
      });

    // ── Non-streaming path ────────────────────────────────────────────────────
    } else {
      const choices = nimResponse.data.choices.map(choice => {
        let finalContent = choice.message?.content || '';

        if (SHOW_REASONING && choice.message?.reasoning_content) {
          finalContent =
            '<think>\n' +
            choice.message.reasoning_content +
            '\n</think>\n\n' +
            finalContent;
        }

        return {
          index:         choice.index,
          message:       { role: choice.message.role, content: finalContent },
          finish_reason: choice.finish_reason
        };
      });

      res.json({
        id:      `chatcmpl-${Date.now()}`,
        object:  'chat.completion',
        created: Math.floor(Date.now() / 1000),
        model:   model,
        choices,
        usage: nimResponse.data.usage || {
          prompt_tokens:     0,
          completion_tokens: 0,
          total_tokens:      0
        }
      });
    }

  } catch (error) {
    console.error('Proxy error:', error.message);
    if (error.response?.data) console.error('NIM error body:', error.response.data);

    res.status(error.response?.status || 500).json({
      error: {
        message: error.message || 'Internal server error',
        type:    'invalid_request_error',
        code:    error.response?.status || 500
      }
    });
  }
});

// ─── 404 catch-all ────────────────────────────────────────────────────────────
app.all('*', (req, res) => {
  res.status(404).json({
    error: {
      message: `Endpoint ${req.path} not found`,
      type:    'invalid_request_error',
      code:    404
    }
  });
});

// ─── Start ────────────────────────────────────────────────────────────────────
app.listen(PORT, () => {
  console.log(`✅ OpenAI → NVIDIA NIM Proxy running on port ${PORT}`);
  console.log(`🔍 Health:    http://localhost:${PORT}/health`);
  console.log(`🧠 Reasoning: ${SHOW_REASONING ? 'SHOWN in <think> tags' : 'HIDDEN'}`);
  console.log(`⚙️  Thinking models: ${THINKING_REQUIRED_MODELS.join(', ')}`);
});  'gpt-4': 'nvidia/llama-3.1-nemotron-ultra-253b-v1',        // 🎭 Best character consistency
  'claude-3-opus': 'nvidia/llama-3.1-nemotron-ultra-253b-v1', // 🎭 Same reliability
  
  // High quality with good memory
  'gpt-4o': 'moonshotai/kimi-k2-thinking',                   // 📝 256k context, native reasoning
  'gemini-pro': 'nvidia/nemotron-3-ultra-550b-a55b',
  'gemini-1.5-pro': 'z-ai/glm4.7',// 📝 Very capable, good memory
  'gemini-1.6-pro': 'stepfun-ai/step-3.5-flash',
  'gemini-1.7-pro': 'google/gemma-4-31b-it', 
  'gemini-1.8-pro': 'deepseek-ai/deepseek-v4-pro',
  'gemini-1.9-pro': 'z-ai/glm-5.1', 
  

  // Fast but still good quality  
  'gpt-4o-mini': 'meta/llama-3.1-70b-instruct',              // ⚡ Fast, decent memory
  'claude-3-sonnet': 'meta/llama-3.1-70b-instruct',          // ⚡ Alternative fast option
  
  // Ultra-fast for testing
  'gpt-3.5-turbo': 'meta/llama-3.1-8b-instruct',
  'gemini-2.6-pro': 'mistralai/mistral-large-3-675b-instruct-2512',// ⚡⚡⚡ Fastest
  
  // Thinking models (slow but very smart)
  'o1': 'deepseek-ai/deepseek-v3.2', // 🧠 685B reasoning model
  'o1-mini': 'deepseek-ai/deepseek-r1-distill-qwen-32b'     // 🧠 Faster thinking
};

// Health check endpoint
app.get('/health', (req, res) => {
  res.json({ 
    status: 'ok', 
    service: 'OpenAI to NVIDIA NIM Proxy', 
    reasoning_display: SHOW_REASONING,
    thinking_models: THINKING_REQUIRED_MODELS
  });
});

// List models endpoint (OpenAI compatible)
app.get('/v1/models', (req, res) => {
  const models = Object.keys(MODEL_MAPPING).map(model => ({
    id: model,
    object: 'model',
    created: Date.now(),
    owned_by: 'nvidia-nim-proxy'
  }));
  
  res.json({
    object: 'list',
    data: models
  });
});

// Chat completions endpoint (main proxy)
app.post('/v1/chat/completions', async (req, res) => {
  try {
    const { model, messages, temperature, max_tokens, stream } = req.body;
    
    // Smart model selection with fallback
    let nimModel = MODEL_MAPPING[model];
    if (!nimModel) {
      try {
        await axios.post(`${NIM_API_BASE}/chat/completions`, {
          model: model,
          messages: [{ role: 'user', content: 'test' }],
          max_tokens: 1
        }, {
          headers: { 'Authorization': `Bearer ${NIM_API_KEY}`, 'Content-Type': 'application/json' },
          validateStatus: (status) => status < 500
        }).then(res => {
          if (res.status >= 200 && res.status < 300) {
            nimModel = model;
          }
        });
      } catch (e) {}
      
      if (!nimModel) {
        const modelLower = model.toLowerCase();
        if (modelLower.includes('gpt-4') || modelLower.includes('claude-opus') || modelLower.includes('405b')) {
          nimModel = 'meta/llama-3.1-405b-instruct';
        } else if (modelLower.includes('claude') || modelLower.includes('gemini') || modelLower.includes('70b')) {
          nimModel = 'meta/llama-3.1-70b-instruct';
        } else {
          nimModel = 'meta/llama-3.1-8b-instruct';
        }
      }
    }
    
    // Transform OpenAI request to NIM format
    // Auto-enable thinking for "Endpoint Only" models
    const requiresThinking = THINKING_REQUIRED_MODELS.includes(nimModel);
    
    const nimRequest = {
      model: nimModel,
      messages: messages,
      temperature: temperature || 0.7,
      max_tokens: max_tokens || 20000,
      top_p: 0.95,
      frequency_penalty: 0.0,
      presence_penalty: 0.0,
      stop: null,
      chat_template_kwargs: {thinking:true, clear_thinking:true, do_sample:true, enable_thinking:true, clear_thinking:true},
      stream: stream || true
    };
    
    // Make request to NVIDIA NIM API
    const response = await axios.post(`${NIM_API_BASE}/chat/completions`, nimRequest, {
      headers: {
        'Authorization': `Bearer ${NIM_API_KEY}`,
        'Content-Type': 'application/json'
      },
      responseType: stream ? 'stream' : 'json'
    });
    
    if (stream) {
      // Handle streaming response with reasoning
      res.setHeader('Content-Type', 'text/event-stream');
      res.setHeader('Cache-Control', 'no-cache');
      res.setHeader('Connection', 'keep-alive');
      
      let buffer = '';
      let reasoningStarted = false;
      
      response.data.on('data', (chunk) => {
        buffer += chunk.toString();
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';
        
        lines.forEach(line => {
          if (line.startsWith('data: ')) {
            if (line.includes('[DONE]')) {
              res.write(line + '\n');
              return;
            }
            
            try {
              const data = JSON.parse(line.slice(6));
              if (data.choices?.[0]?.delta) {
                const reasoning = data.choices[0].delta.reasoning_content;
                const content = data.choices[0].delta.content;
                
                if (SHOW_REASONING) {
                  let combinedContent = '';
                  
                  if (reasoning && !reasoningStarted) {
                    combinedContent = '<think>\n' + reasoning;
                    reasoningStarted = true;
                  } else if (reasoning) {
                    combinedContent = reasoning;
                  }
                  
                  if (content && reasoningStarted) {
                    combinedContent += '</think>\n\n' + content;
                    reasoningStarted = false;
                  } else if (content) {
                    combinedContent += content;
                  }
                  
                  if (combinedContent) {
                    data.choices[0].delta.content = combinedContent;
                    delete data.choices[0].delta.reasoning_content;
                  }
                } else {
                  if (content) {
                    data.choices[0].delta.content = content;
                  } else {
                    data.choices[0].delta.content = '';
                  }
                  delete data.choices[0].delta.reasoning_content;
                }
              }
              res.write(`data: ${JSON.stringify(data)}\n\n`);
            } catch (e) {
              res.write(line + '\n');
            }
          }
        });
      });
      
      response.data.on('end', () => res.end());
      response.data.on('error', (err) => {
        console.error('Stream error:', err);
        res.end();
      });
    } else {
      // Transform NIM response to OpenAI format with reasoning
      const openaiResponse = {
        id: `chatcmpl-${Date.now()}`,
        object: 'chat.completion',
        created: Math.floor(Date.now() / 1000),
        model: model,
        choices: response.data.choices.map(choice => {
          let fullContent = choice.message?.content || '';
          
          if (SHOW_REASONING && choice.message?.reasoning_content) {
            fullContent = '<think>\n' + choice.message.reasoning_content + '\n</think>\n\n' + fullContent;
          }
          
          return {
            index: choice.index,
            message: {
              role: choice.message.role,
              content: fullContent
            },
            finish_reason: choice.finish_reason
          };
        }),
        usage: response.data.usage || {
          prompt_tokens: 0,
          completion_tokens: 0,
          total_tokens: 0
        }
      };
      
      res.json(openaiResponse);
    }
    
  } catch (error) {
    console.error('Proxy error:', error.message);
    
    res.status(error.response?.status || 500).json({
      error: {
        message: error.message || 'Internal server error',
        type: 'invalid_request_error',
        code: error.response?.status || 500
      }
    });
  }
});

// Catch-all for unsupported endpoints
app.all('*', (req, res) => {
  res.status(404).json({
    error: {
      message: `Endpoint ${req.path} not found`,
      type: 'invalid_request_error',
      code: 404
    }
  });
});

app.listen(PORT, () => {
  console.log(`OpenAI to NVIDIA NIM Proxy running on port ${PORT}`);
  console.log(`Health check: http://localhost:${PORT}/health`);
  console.log(`Reasoning display: ${SHOW_REASONING ? 'ENABLED' : 'DISABLED'}`);
  console.log(`Thinking models: ${THINKING_REQUIRED_MODELS.join(', ')}`);
});
