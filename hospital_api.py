# hospital_api.py - Flask API for Hospital Resource Management Dashboard
"""
RESTful API backend for the hospital dashboard
Provides endpoints for real-time analysis and chat functionality
"""

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from hospital_resource_manager_llm import hospital_agent
import json
import os
from datetime import datetime
import glob

app = Flask(__name__, static_folder='static')
CORS(app)

@app.route('/')
def index():
    """Serve the main dashboard"""
    return send_from_directory('static', 'index.html')

@app.route('/api/status', methods=['GET'])
def get_status():
    """Get current hospital status with AI insights"""
    try:
        result = hospital_agent.invoke({"task_type": "full_analysis"})

        response = {
            'timestamp': datetime.now().isoformat(),
            'aqi': result.get('aqi_data', {}),
            'health': result.get('health_predictions', {}),
            'capacity': result.get('bed_capacity_plan', {}),
            'resources': result.get('resource_allocation', {}),
            'festivals': result.get('festival_predictions', {}),
            'alerts': result.get('critical_alerts', []),
            'ai_insights': result.get('llm_insights', {}),
            'epidemic': result.get('epidemic_alerts', [])
        }

        return jsonify(response)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/chat', methods=['POST'])
def chat():
    """AI chat interface for administrators"""
    try:
        data = request.json
        user_query = data.get('query', '')

        if not user_query:
            return jsonify({'error': 'No query provided'}), 400

        result = hospital_agent.invoke({
            "task_type": "user_query",
            "user_query": user_query
        })

        # Extract AI response from insights
        insights = result.get('llm_insights', {})
        response_text = insights.get('summary', 'No response generated')

        return jsonify({
            'query': user_query,
            'response': response_text,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/reports', methods=['GET'])
def get_reports():
    """Get list of recent reports"""
    try:
        reports_dir = 'reports'
        if not os.path.exists(reports_dir):
            return jsonify({'reports': []})

        # Get all JSON reports
        reports = []
        for file in glob.glob(os.path.join(reports_dir, '*.json')):
            filename = os.path.basename(file)
            file_time = os.path.getmtime(file)
            reports.append({
                'filename': filename,
                'timestamp': datetime.fromtimestamp(file_time).isoformat(),
                'type': filename.split('_')[0]
            })

        # Sort by timestamp, newest first
        reports.sort(key=lambda x: x['timestamp'], reverse=True)

        return jsonify({'reports': reports[:10]})  # Last 10 reports
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/reports/<filename>', methods=['GET'])
def get_report(filename):
    """Get specific report content"""
    try:
        filepath = os.path.join('reports', filename)
        if not os.path.exists(filepath):
            return jsonify({'error': 'Report not found'}), 404

        with open(filepath, 'r', encoding='utf-8') as f:
            content = json.load(f)

        return jsonify(content)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Create static folder if it doesn't exist
    os.makedirs('static', exist_ok=True)
    print("\n🚀 Starting Hospital Resource Management API...")
    print("📊 Dashboard: http://localhost:5000")
    print("🔌 API Endpoints:")
    print("   GET  /api/status - Current hospital status")
    print("   POST /api/chat - AI chat interface")
    print("   GET  /api/reports - List recent reports")
    print("\n✅ Server ready!\n")
    app.run(debug=True, host='0.0.0.0', port=5000)
