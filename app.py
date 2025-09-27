#!/usr/bin/env python3
"""
Web application wrapper for NLP CLI tools
Provides REST API endpoints for the various NLP functionalities
"""

import os
import json
import tempfile
import subprocess
from flask import Flask, request, jsonify, render_template_string
from werkzeug.utils import secure_filename
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# HTML template for the web interface
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>NLP Toolkit</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; background-color: #f5f5f5; }
        .container { max-width: 800px; margin: 0 auto; background: white; padding: 30px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        h1 { color: #333; text-align: center; }
        .section { margin: 30px 0; padding: 20px; border: 1px solid #ddd; border-radius: 5px; }
        .section h2 { margin-top: 0; color: #555; }
        textarea { width: 100%; height: 100px; padding: 10px; border: 1px solid #ddd; border-radius: 4px; }
        button { background: #007bff; color: white; padding: 10px 20px; border: none; border-radius: 4px; cursor: pointer; margin: 5px; }
        button:hover { background: #0056b3; }
        .result { background: #f8f9fa; padding: 15px; border-radius: 4px; margin-top: 10px; white-space: pre-wrap; }
        .error { background: #f8d7da; color: #721c24; }
        .success { background: #d4edda; color: #155724; }
    </style>
</head>
<body>
    <div class="container">
        <h1>NLP Toolkit</h1>
        
        <div class="section">
            <h2>Text Processing</h2>
            <textarea id="textInput" placeholder="Enter your text here..."></textarea>
            <br>
            <button onclick="processText('regex', 'dates')">Extract Dates</button>
            <button onclick="processText('regex', 'abbr')">Extract Abbreviations</button>
            <button onclick="processText('regex', 'html')">Clean HTML</button>
            <button onclick="processText('regex', 'json')">Clean JSON</button>
            <div id="regexResult" class="result" style="display:none;"></div>
        </div>

        <div class="section">
            <h2>Tokenization</h2>
            <button onclick="runTokenizers()">Run Tokenizers Comparison</button>
            <div id="tokenizerResult" class="result" style="display:none;"></div>
        </div>

        <div class="section">
            <h2>Sequence-to-Sequence</h2>
            <button onclick="runSeq2Seq()">Run Seq2Seq Training/Evaluation</button>
            <div id="seq2seqResult" class="result" style="display:none;"></div>
        </div>

        <div class="section">
            <h2>POS Tagging & NER</h2>
            <button onclick="runPosNer()">Run POS Tagging & NER</button>
            <div id="posNerResult" class="result" style="display:none;"></div>
        </div>
    </div>

    <script>
        async function processText(mode, subMode) {
            const text = document.getElementById('textInput').value;
            if (!text.trim()) {
                alert('Please enter some text');
                return;
            }
            
            const resultDiv = document.getElementById('regexResult');
            resultDiv.style.display = 'block';
            resultDiv.textContent = 'Processing...';
            
            try {
                const response = await fetch('/api/regex', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ mode: subMode, text: text })
                });
                const data = await response.json();
                resultDiv.textContent = JSON.stringify(data, null, 2);
                resultDiv.className = response.ok ? 'result success' : 'result error';
            } catch (error) {
                resultDiv.textContent = 'Error: ' + error.message;
                resultDiv.className = 'result error';
            }
        }

        async function runTokenizers() {
            const resultDiv = document.getElementById('tokenizerResult');
            resultDiv.style.display = 'block';
            resultDiv.textContent = 'Running tokenizers...';
            
            try {
                const response = await fetch('/api/tokenizers', { method: 'POST' });
                const data = await response.json();
                resultDiv.textContent = JSON.stringify(data, null, 2);
                resultDiv.className = response.ok ? 'result success' : 'result error';
            } catch (error) {
                resultDiv.textContent = 'Error: ' + error.message;
                resultDiv.className = 'result error';
            }
        }

        async function runSeq2Seq() {
            const resultDiv = document.getElementById('seq2seqResult');
            resultDiv.style.display = 'block';
            resultDiv.textContent = 'Running seq2seq...';
            
            try {
                const response = await fetch('/api/seq2seq', { method: 'POST' });
                const data = await response.json();
                resultDiv.textContent = JSON.stringify(data, null, 2);
                resultDiv.className = response.ok ? 'result success' : 'result error';
            } catch (error) {
                resultDiv.textContent = 'Error: ' + error.message;
                resultDiv.className = 'result error';
            }
        }

        async function runPosNer() {
            const resultDiv = document.getElementById('posNerResult');
            resultDiv.style.display = 'block';
            resultDiv.textContent = 'Running POS tagging & NER...';
            
            try {
                const response = await fetch('/api/pos-ner', { method: 'POST' });
                const data = await response.json();
                resultDiv.textContent = JSON.stringify(data, null, 2);
                resultDiv.className = response.ok ? 'result success' : 'result error';
            } catch (error) {
                resultDiv.textContent = 'Error: ' + error.message;
                resultDiv.className = 'result error';
            }
        }
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    """Serve the main web interface"""
    return render_template_string(HTML_TEMPLATE)

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({"status": "healthy", "service": "nlp-toolkit"})

@app.route('/api/regex', methods=['POST'])
def regex_processing():
    """Handle regex-based text processing"""
    try:
        data = request.get_json()
        mode = data.get('mode', 'dates')
        text = data.get('text', '')
        
        if not text:
            return jsonify({"error": "No text provided"}), 400
        
        # Create temporary file with input text
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(text)
            temp_file = f.name
        
        try:
            # Run the regex CLI tool
            cmd = ['python', '-m', 'src.cli.run_regex', '--mode', mode, '--text', text]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                return jsonify({
                    "success": True,
                    "mode": mode,
                    "input_text": text,
                    "output": result.stdout
                })
            else:
                return jsonify({
                    "success": False,
                    "error": result.stderr
                }), 500
                
        finally:
            # Clean up temporary file
            if os.path.exists(temp_file):
                os.unlink(temp_file)
                
    except subprocess.TimeoutExpired:
        return jsonify({"error": "Processing timeout"}), 408
    except Exception as e:
        logger.error(f"Error in regex processing: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/tokenizers', methods=['POST'])
def tokenizers():
    """Handle tokenizer comparison"""
    try:
        cmd = ['python', '-m', 'src.cli.run_tokenizers', '--cfg', 'configs/tokenization.yml', '--compare']
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        
        if result.returncode == 0:
            return jsonify({
                "success": True,
                "output": result.stdout
            })
        else:
            return jsonify({
                "success": False,
                "error": result.stderr
            }), 500
            
    except subprocess.TimeoutExpired:
        return jsonify({"error": "Processing timeout"}), 408
    except Exception as e:
        logger.error(f"Error in tokenizers: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/seq2seq', methods=['POST'])
def seq2seq():
    """Handle sequence-to-sequence processing"""
    try:
        cmd = [
            'python', '-m', 'src.cli.run_seq2seq',
            '--prep_cfg', 'configs/preprocess.yml',
            '--model_cfg', 'configs/model_seq2seq.yml'
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            return jsonify({
                "success": True,
                "output": result.stdout
            })
        else:
            return jsonify({
                "success": False,
                "error": result.stderr
            }), 500
            
    except subprocess.TimeoutExpired:
        return jsonify({"error": "Processing timeout"}), 408
    except Exception as e:
        logger.error(f"Error in seq2seq: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/pos-ner', methods=['POST'])
def pos_ner():
    """Handle POS tagging and NER"""
    try:
        cmd = ['python', '-m', 'src.cli.run_pos_ner']
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        
        if result.returncode == 0:
            return jsonify({
                "success": True,
                "output": result.stdout
            })
        else:
            return jsonify({
                "success": False,
                "error": result.stderr
            }), 500
            
    except subprocess.TimeoutExpired:
        return jsonify({"error": "Processing timeout"}), 408
    except Exception as e:
        logger.error(f"Error in POS/NER: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=False)
