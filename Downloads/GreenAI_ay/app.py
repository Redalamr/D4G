from flask import Flask, request, jsonify, render_template
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM
import torch
import time
from codecarbon import EmissionsTracker
import os

app = Flask(__name__)

# Set seeds for reproducibility
os.environ['PYTHONHASHSEED'] = '0'
torch.manual_seed(42)

# Model configurations
MODEL_CONFIGS = {
    'pythia-70m': {
        'name': 'EleutherAI/pythia-70m-deduped',
        'type': 'causal',
    },
    'flan-t5-small': {
        'name': 'google/flan-t5-small',
        'type': 'seq2seq',
    },
    'flan-t5-base': {
        'name': 'google/flan-t5-base',
        'type': 'seq2seq',
    }
}

# Store loaded models (lazy loading)
loaded_models = {}

print("Backend initialized. Models will be loaded on first use.")

def load_model(model_key, optimized=False):
    """Load and cache a model"""
    cache_key = f"{model_key}_{'opt' if optimized else 'base'}"

    if cache_key in loaded_models:
        print(f"Using cached model: {cache_key}")
        return loaded_models[cache_key]

    config = MODEL_CONFIGS[model_key]
    model_name = config['name']
    model_type = config['type']

    print(f"Loading model: {model_name} (optimized={optimized})")

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if model_type == 'causal':
        model = AutoModelForCausalLM.from_pretrained(model_name)
    else:  # seq2seq
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    # Apply quantization if optimized
    if optimized:
        model = torch.quantization.quantize_dynamic(
            model, {torch.nn.Linear}, dtype=torch.qint8
        )

    loaded_models[cache_key] = {
        'tokenizer': tokenizer,
        'model': model,
        'type': model_type
    }

    print(f"Model loaded: {cache_key}")
    return loaded_models[cache_key]

def generate_summary(text, model_key='flan-t5-base', optimized=False):
    """Generate a 10-15 word summary in French"""

    # Load the model
    model_data = load_model(model_key, optimized)
    tokenizer = model_data['tokenizer']
    model = model_data['model']
    model_type = model_data['type']

    # Measure energy and time
    tracker = EmissionsTracker(save_to_file=False, logging_logger=None)
    tracker.start()
    start_time = time.time()

    if model_type == 'seq2seq':
        # T5/FLAN-T5 models - use instruction format
        prompt = f"Résume ce texte en 10-15 mots en français: {text[:2000]}"
        inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)

        with torch.no_grad():
            outputs = model.generate(
                inputs.input_ids,
                max_length=30,
                min_length=10,
                do_sample=False,
                num_beams=4,
                early_stopping=True
            )

        summary = tokenizer.decode(outputs[0], skip_special_tokens=True)

    else:
        # Causal LM (pythia) - use completion format
        text_truncated = text[:500]
        prompt = f"Texte: {text_truncated}\nEn résumé:"
        inputs = tokenizer(prompt, return_tensors="pt", max_length=200, truncation=True)

        with torch.no_grad():
            outputs = model.generate(
                inputs.input_ids,
                max_new_tokens=25,
                min_new_tokens=8,
                do_sample=True,
                temperature=0.8,
                top_k=50,
                top_p=0.95,
                repetition_penalty=1.2,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id
            )

        full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Extract only newly generated text
        if "En résumé:" in full_text:
            summary = full_text.split("En résumé:")[-1].strip()
        else:
            summary = full_text[len(prompt):].strip()

        summary = summary.split('\n')[0].strip()
        summary = summary.split('.')[0].strip()

    # Stop measurements
    latency = (time.time() - start_time) * 1000  # Convert to ms
    emissions = tracker.stop()
    energy_wh = emissions * 1000 if emissions else 0  # Convert to Wh

    print("\n=== DEBUG ===")
    print(f"Model: {model_key}, Type: {model_type}, Optimized: {optimized}")
    print(f"Summary: {summary}")
    print(f"Word count: {len(summary.split())}")
    print(f"Energy: {energy_wh} Wh, Latency: {latency} ms")
    print("=============\n")

    # Ensure word count is reasonable (10-15 words)
    words = summary.split()
    if len(words) > 15:
        summary = ' '.join(words[:15])
    elif len(words) < 5:
        # Fallback if summary too short
        text_words = text.split()[:12]
        summary = ' '.join(text_words) + "..."

    return summary, energy_wh, latency

@app.route('/')
def index():
    """Serve the web interface"""
    return render_template('index.html')

@app.route('/summarize', methods=['POST'])
def summarize():
    """API endpoint for text summarization"""
    try:
        data = request.get_json()
        text = data.get('text', '')
        optimized = data.get('optimized', False)
        model = data.get('model', 'flan-t5-base')  # Default to flan-t5-base

        if not text:
            return jsonify({'error': 'No text provided'}), 400

        # Validate model
        if model not in MODEL_CONFIGS:
            return jsonify({'error': f'Invalid model: {model}'}), 400

        # Generate summary
        summary, energy_wh, latency = generate_summary(text, model, optimized)

        return jsonify({
            'summary': summary,
            'energy_wh': round(energy_wh, 6),
            'latency_ms': round(latency, 2),
            'mode': 'optimized' if optimized else 'baseline',
            'model': model
        })

    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8080, debug=False)
