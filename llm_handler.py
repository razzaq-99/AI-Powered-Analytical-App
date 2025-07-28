"""
LLM Handler for Ollama Integration with LangChain
"""

from langchain.llms import Ollama
from langchain.llms.base import LLM
from langchain.schema import BaseMessage, HumanMessage, SystemMessage
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferWindowMemory
from typing import Optional, List, Dict, Any
import json
import logging
from config_py import Config

logger = logging.getLogger(__name__)

class OllamaLLM(LLM):
    """Custom Ollama LLM wrapper for LangChain"""
    
    model: str = Config.OLLAMA_MODEL
    base_url: str = Config.OLLAMA_BASE_URL

    def __init__(self, model: str = None, **kwargs):
        super().__init__(**kwargs)
        if model:
            self.model = model

    @property
    def _llm_type(self) -> str:
        return "ollama"

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        try:
            ollama = Ollama(model=self.model)
            response = ollama.generate(prompt=prompt)
            return response['text']
        except Exception as e:
            logger.error(f"Error calling Ollama: {str(e)}")
            return f"Error: Could not generate response - {str(e)}"

class LLMHandler:
    def __init__(self, model_name: str = Config.OLLAMA_MODEL):
        self.model_name = model_name
        self.llm = OllamaLLM(model=model_name)
        self.conversations = {}
        self.system_prompts = Config.SYSTEM_PROMPTS
        self.analysis_context = {}
        self._init_conversation_chains()

    def _init_conversation_chains(self):
        for role in ['analyzer', 'decision_maker', 'chat_assistant']:
            memory = ConversationBufferWindowMemory(k=10)
            self.conversations[role] = ConversationChain(
                llm=self.llm,
                memory=memory,
                verbose=False
            )

    def check_ollama_connection(self) -> bool:
        try:
            models = Ollama.list()
            available_models = [model['name'] for model in models['models']]
            if self.model_name not in available_models:
                logger.warning(f"Model {self.model_name} not found. Available: {available_models}")
                self.pull_model(self.model_name)
            return True
        except Exception as e:
            logger.error(f"Ollama connection failed: {str(e)}")
            return False

    def pull_model(self, model_name: str) -> bool:
        try:
            logger.info(f"Pulling model {model_name}...")
            Ollama.pull(model_name)
            logger.info(f"Successfully pulled model {model_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to pull model {model_name}: {str(e)}")
            return False

    def get_data_cleaning_suggestions(self, data_summary: Dict) -> str:
        prompt = f"""
{self.system_prompts['analyzer']}

Based on the following data summary, provide specific data cleaning recommendations:

Data Summary:
{json.dumps(data_summary, indent=2)}

Please analyze:
1. Missing values and how to handle them
2. Outliers detection and treatment
3. Data type corrections needed
4. Duplicate records handling
5. Column standardization requirements

Provide actionable recommendations with reasoning.
"""
        try:
            return self.conversations['analyzer'].predict(input=prompt)
        except Exception as e:
            logger.error(f"Error getting cleaning suggestions: {str(e)}")
            return "Error generating cleaning suggestions"

    def analyze_data_insights(self, cleaned_data_summary: Dict, viz_insights: List[str]) -> str:
        prompt = f"""
{self.system_prompts['analyzer']}

Perform comprehensive analysis on the following cleaned dataset:

Data Summary:
{json.dumps(cleaned_data_summary, indent=2)}

Visualization Insights:
{chr(10).join(viz_insights)}

Please provide:
1. Key statistical insights
2. Pattern identification
3. Correlation analysis
4. Trend analysis
5. Anomaly detection
6. Business implications
7. Actionable recommendations

Structure your response with clear sections and bullet points.
"""
        try:
            response = self.conversations['analyzer'].predict(input=prompt)
            self.analysis_context['insights'] = response
            return response
        except Exception as e:
            logger.error(f"Error analyzing insights: {str(e)}")
            return "Error generating insights"

    def make_data_driven_decisions(self, analysis_results: str, business_context: str = "") -> str:
        prompt = f"""
{self.system_prompts['decision_maker']}

Based on the following analysis, provide strategic decisions and recommendations:

Analysis Results:
{analysis_results}

Business Context:
{business_context}

Please provide:
1. Strategic recommendations
2. Risk assessment
3. Opportunity identification
4. Implementation priorities
5. Success metrics
6. Next steps
"""
        try:
            response = self.conversations['decision_maker'].predict(input=prompt)
            self.analysis_context['decisions'] = response
            return response
        except Exception as e:
            logger.error(f"Error making decisions: {str(e)}")
            return "Error generating decisions"

    def chat_with_analysis(self, user_query: str, context_data: Dict = None) -> str:
        analysis_context = self.analysis_context.get('insights', 'No analysis available')
        decision_context = self.analysis_context.get('decisions', 'No decisions made')

        context_info = ""
        if context_data:
            context_info = f"\nData Context:\n{json.dumps(context_data, indent=2)}"

        prompt = f"""
{self.system_prompts['chat_assistant']}

Previous Analysis Context:
{analysis_context}

Previous Decisions:
{decision_context}
{context_info}

User Query: {user_query}

Provide a helpful, accurate response based on the analysis context.
"""
        try:
            return self.conversations['chat_assistant'].predict(input=prompt)
        except Exception as e:
            logger.error(f"Error in chat response: {str(e)}")
            return "I apologize, but I encountered an error processing your query."

    def generate_chart_insights(self, chart_data: Dict, chart_type: str) -> str:
        prompt = f"""
Analyze the following {chart_type} chart and provide key insights:

Chart Data Summary:
{json.dumps(chart_data, indent=2)}

Please provide:
1. What the chart reveals about the data
2. Key patterns or trends visible
3. Notable outliers or anomalies
4. Business implications
5. Recommended actions

Keep the response concise but insightful.
"""
        try:
            return self.llm._call(prompt)
        except Exception as e:
            logger.error(f"Error generating chart insights: {str(e)}")
            return f"Unable to generate insights for {chart_type}"

    def get_analysis_summary(self) -> Dict[str, str]:
        return {
            'insights': self.analysis_context.get('insights', 'No insights generated'),
            'decisions': self.analysis_context.get('decisions', 'No decisions made'),
            'status': 'Analysis completed' if self.analysis_context else 'No analysis performed'
        }

    def clear_context(self):
        self.analysis_context = {}
        for chain in self.conversations.values():
            chain.memory.clear()
        logger.info("Analysis context and conversation memory cleared")

    def set_analysis_context(self, context: Dict):
        self.analysis_context.update(context)

    def generate_executive_summary(self, full_analysis: Dict) -> str:
        prompt = f"""
{self.system_prompts['analyzer']}

Create an executive summary based on the complete analysis:

Full Analysis:
{json.dumps(full_analysis, indent=2)}

Provide:
1. Key findings (3-5 bullet points)
2. Critical insights
3. Main recommendations
4. Business impact
5. Next steps

Keep it concise and executive-level appropriate.
"""
        try:
            return self.llm._call(prompt)
        except Exception as e:
            logger.error(f"Error generating executive summary: {str(e)}")
            return "Error generating executive summary"

    def validate_model_response(self, response: str) -> bool:
        if not response or len(response.strip()) < 10:
            return False
        if "error" in response.lower() and "could not generate" in response.lower():
            return False
        return True
