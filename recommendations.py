"""
Decision Engine Module - AI-powered decision making based on analysis
"""
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np
from langchain.schema import HumanMessage, SystemMessage
import logging
import json

logger = logging.getLogger(__name__)

class DecisionEngine:
    def __init__(self, llm_handler):
        self.llm_handler = llm_handler
        self.decision_history = []
        self.recommendations = {}
    
    def analyze_business_impact(self, analysis_results: Dict[str, Any], 
                              df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze business impact using AI-powered insights"""
        
        data_summary = self._prepare_data_summary(df)
        analysis_summary = self._prepare_analysis_summary(analysis_results, df)
        
        business_impact = {
            'data_quality_impact': self._assess_data_quality_impact_ai(data_summary, analysis_results),
            'operational_risks': self._identify_operational_risks_ai(data_summary, analysis_results),
            'opportunities': self._identify_opportunities_ai(data_summary, analysis_results),
            'priority_actions': self._prioritize_actions_ai(data_summary, analysis_results)
        }
        
        return business_impact
    
    def _prepare_data_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Prepare comprehensive data summary for AI analysis"""
        
        missing_pct = (df.isnull().sum() / len(df) * 100)
        
        summary = {
            'dataset_shape': {
                'rows': len(df),
                'columns': len(df.columns)
            },
            'data_types': df.dtypes.value_counts().to_dict(),
            'missing_data': {
                'columns_with_missing': missing_pct[missing_pct > 0].to_dict(),
                'average_missing_percentage': missing_pct.mean(),
                'columns_high_missing': missing_pct[missing_pct > 20].index.tolist()
            },
            'data_quality_metrics': {
                'completeness_score': ((1 - df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100),
                'duplicate_rows': df.duplicated().sum(),
                'unique_values_per_column': {col: df[col].nunique() for col in df.columns}
            },
            'statistical_overview': self._get_statistical_overview(df)
        }
        
        return summary
    
    def _get_statistical_overview(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get statistical overview of the dataset"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        categorical_cols = df.select_dtypes(include=['object']).columns
        
        overview = {
            'numeric_features': len(numeric_cols),
            'categorical_features': len(categorical_cols),
            'numeric_stats': {},
            'categorical_stats': {}
        }
        
        if len(numeric_cols) > 0:
            numeric_df = df[numeric_cols]
            overview['numeric_stats'] = {
                'mean_values': numeric_df.mean().to_dict(),
                'std_values': numeric_df.std().to_dict(),
                'skewness': numeric_df.skew().to_dict(),
                'potential_outliers': self._detect_outliers_summary(numeric_df)
            }
        
        
        if len(categorical_cols) > 0:
            overview['categorical_stats'] = {
                col: {
                    'unique_count': df[col].nunique(),
                    'most_frequent': df[col].mode().iloc[0] if not df[col].empty else None,
                    'concentration_ratio': (df[col].value_counts().iloc[0] / len(df)) * 100 if not df[col].empty else 0
                } for col in categorical_cols[:5]  
            }
        
        return overview
    
    def _detect_outliers_summary(self, numeric_df: pd.DataFrame) -> Dict[str, int]:
        """Detect outliers summary for numeric columns"""
        outliers_summary = {}
        
        for col in numeric_df.columns:
            Q1 = numeric_df[col].quantile(0.25)
            Q3 = numeric_df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = numeric_df[(numeric_df[col] < lower_bound) | (numeric_df[col] > upper_bound)]
            outliers_summary[col] = len(outliers)
        
        return outliers_summary
    
    def _assess_data_quality_impact_ai(self, data_summary: Dict[str, Any], 
                                      analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Use AI to assess data quality impact on business"""
        
        prompt = f"""
        As a data quality expert, analyze the following dataset summary and provide specific business impact assessment:

        Dataset Summary:
        {json.dumps(data_summary, indent=2)}

        Analysis Results Context:
        {json.dumps(analysis_results, indent=2)}

        Please assess:
        1. How data quality issues impact business operations
        2. Which columns/issues are most critical for business decisions
        3. Potential revenue/cost implications of data quality problems
        4. Specific recommendations to improve data quality

        Provide response in the following JSON format:
        {{
            "critical_issues": [list of most critical data quality problems],
            "business_impact_assessment": "detailed description of business impact",
            "severity_level": "High/Medium/Low",
            "estimated_impact": "quantified impact if possible",
            "immediate_actions": [list of immediate actions needed],
            "recommendations": [list of specific recommendations]
        }}
        """
        
        try:
            response = self.llm_handler.llm._call(prompt)
            try:
                return json.loads(response)
            except json.JSONDecodeError:
                return {
                    "assessment": response,
                    "source": "ai_generated"
                }
        except Exception as e:
            logger.error(f"Error in AI data quality assessment: {str(e)}")
            return {"error": "Could not generate AI assessment", "fallback": self._fallback_data_quality_assessment(data_summary)}
    
    def _identify_operational_risks_ai(self, data_summary: Dict[str, Any], 
                                      analysis_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Use AI to identify operational risks from data patterns"""
        
        prompt = f"""
        As a business risk analyst, analyze the following data and identify operational risks:

        Dataset Summary:
        {json.dumps(data_summary, indent=2)}

        Analysis Context:
        {json.dumps(analysis_results, indent=2)}

        Identify operational risks considering:
        1. Data reliability risks
        2. Business continuity risks
        3. Decision-making risks due to data issues
        4. Compliance and regulatory risks
        5. Operational efficiency risks

        For each risk, provide:
        - Risk type and description
        - Severity level (Critical/High/Medium/Low)
        - Potential business impact
        - Likelihood of occurrence
        - Recommended mitigation strategies

        Provide response as a JSON array of risk objects.
        """
        
        try:
            response = self.llm_handler.llm._call(prompt)
            try:
                risks = json.loads(response)
                return risks if isinstance(risks, list) else [{"description": response, "source": "ai_generated"}]
            except json.JSONDecodeError:
                return [{"description": response, "source": "ai_generated", "type": "AI Analysis"}]
        except Exception as e:
            logger.error(f"Error in AI risk identification: {str(e)}")
            return self._fallback_risk_identification(data_summary)
    
    def _identify_opportunities_ai(self, data_summary: Dict[str, Any], 
                                  analysis_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Use AI to identify business opportunities from data insights"""
        
        prompt = f"""
        As a business strategist, analyze the following data to identify growth opportunities:

        Dataset Summary:
        {json.dumps(data_summary, indent=2)}

        Analysis Results:
        {json.dumps(analysis_results, indent=2)}

        Identify opportunities in:
        1. Market expansion possibilities
        2. Process optimization opportunities
        3. Customer segmentation and targeting
        4. Product/service improvement areas
        5. Cost reduction opportunities
        6. Revenue enhancement strategies
        7. Competitive advantages

        For each opportunity, provide:
        - Opportunity type and description
        - Priority level (High/Medium/Low)
        - Potential business value
        - Implementation complexity
        - Required resources
        - Expected timeline
        - Success metrics

        Provide response as a JSON array of opportunity objects.
        """
        
        try:
            response = self.llm_handler.llm._call(prompt)
            try:
                opportunities = json.loads(response)
                return opportunities if isinstance(opportunities, list) else [{"description": response, "source": "ai_generated"}]
            except json.JSONDecodeError:
                return [{"description": response, "source": "ai_generated", "type": "AI Analysis"}]
        except Exception as e:
            logger.error(f"Error in AI opportunity identification: {str(e)}")
            return self._fallback_opportunity_identification(data_summary)
    
    def _prioritize_actions_ai(self, data_summary: Dict[str, Any], 
                              analysis_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Use AI to prioritize recommended actions"""
        
        prompt = f"""
        As a business operations consultant, prioritize actions based on the data analysis:

        Dataset Summary:
        {json.dumps(data_summary, indent=2)}

        Analysis Results:
        {json.dumps(analysis_results, indent=2)}

        Create a prioritized action plan considering:
        1. Business impact (revenue, cost, efficiency)
        2. Implementation effort and complexity
        3. Resource requirements
        4. Time to value
        5. Risk mitigation importance
        6. Strategic alignment

        For each action, provide:
        - Action name and description
        - Priority level (Critical/High/Medium/Low)
        - Business justification
        - Implementation effort (High/Medium/Low)
        - Expected impact (High/Medium/Low)
        - Timeline estimate
        - Required resources
        - Success criteria

        Organize actions in priority order and provide as JSON array.
        """
        
        try:
            response = self.llm_handler.llm._call(prompt)
            try:
                actions = json.loads(response)
                return actions if isinstance(actions, list) else [{"description": response, "source": "ai_generated"}]
            except json.JSONDecodeError:
                return [{"description": response, "source": "ai_generated", "type": "AI Prioritization"}]
        except Exception as e:
            logger.error(f"Error in AI action prioritization: {str(e)}")
            return self._fallback_action_prioritization(data_summary)
    
    def generate_strategic_recommendations(self, analysis_results: Dict[str, Any], 
                                         df: pd.DataFrame, context: str = "") -> Dict[str, Any]:
        """Generate comprehensive strategic recommendations using AI"""
        
        data_summary = self._prepare_data_summary(df)
        analysis_summary = self._prepare_analysis_summary(analysis_results, df)
        
        prompt = f"""
        As a senior business consultant and data strategist, provide comprehensive strategic recommendations:

        DATASET OVERVIEW:
        {json.dumps(data_summary, indent=2)}

        ANALYSIS RESULTS:
        {json.dumps(analysis_results, indent=2)}

        BUSINESS CONTEXT:
        {context}

        PROVIDE STRATEGIC RECOMMENDATIONS FOR:

        1. IMMEDIATE ACTIONS (0-30 days):
           - Critical issues requiring immediate attention
           - Quick wins that can be implemented rapidly
           - Risk mitigation priorities

        2. SHORT-TERM STRATEGY (1-6 months):
           - Data infrastructure improvements
           - Process optimization initiatives
           - Team capability building

        3. LONG-TERM VISION (6+ months):
           - Advanced analytics implementation
           - Strategic data initiatives
           - Competitive advantage development

        4. IMPLEMENTATION ROADMAP:
           - Phase-wise implementation plan
           - Resource allocation recommendations
           - Success milestones and KPIs

        5. EXPECTED OUTCOMES:
           - Quantifiable business benefits
           - Risk reduction achievements
           - Performance improvement metrics

        6. INVESTMENT REQUIREMENTS:
           - Technology investments needed
           - Human resource requirements
           - Training and development needs

        Please provide detailed, actionable recommendations that align with business objectives and drive measurable value.
        """
        
        try:
            ai_recommendations = self.llm_handler.llm._call(prompt)
            
            strategic_recommendations = {
                'ai_strategic_analysis': ai_recommendations,
                'business_impact_assessment': self.analyze_business_impact(analysis_results, df),
                'implementation_framework': self._create_implementation_framework_ai(data_summary, analysis_results, context),
                'success_metrics': self._define_success_metrics_ai(data_summary, analysis_results),
                'risk_assessment': self._generate_risk_assessment_ai(data_summary, analysis_results)
            }
            
            return strategic_recommendations
            
        except Exception as e:
            logger.error(f"Error generating strategic recommendations: {str(e)}")
            return {
                'error': 'Could not generate AI recommendations',
                'business_impact': self.analyze_business_impact(analysis_results, df),
                'fallback_recommendations': self._generate_fallback_recommendations(analysis_results, df)
            }
    
    def _create_implementation_framework_ai(self, data_summary: Dict[str, Any], 
                                           analysis_results: Dict[str, Any], context: str) -> Dict[str, Any]:
        """Create AI-powered implementation framework"""
        
        prompt = f"""
        Create a detailed implementation framework for the data-driven initiatives:

        Data Context:
        {json.dumps(data_summary, indent=2)}

        Analysis Results:
        {json.dumps(analysis_results, indent=2)}

        Business Context:
        {context}

        Provide an implementation framework with:
        1. Project phases with timelines
        2. Resource requirements for each phase
        3. Dependencies and prerequisites
        4. Risk mitigation strategies
        5. Success criteria and checkpoints
        6. Budget considerations
        7. Change management recommendations

        Structure as JSON with clear phases and actionable items.
        """
        
        try:
            response = self.llm_handler.llm._call(prompt)
            try:
                return json.loads(response)
            except json.JSONDecodeError:
                return {"framework": response, "source": "ai_generated"}
        except Exception as e:
            logger.error(f"Error creating implementation framework: {str(e)}")
            return {"error": "Could not generate implementation framework"}
    
    def _define_success_metrics_ai(self, data_summary: Dict[str, Any], 
                                  analysis_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Define AI-powered success metrics"""
        
        prompt = f"""
        Define comprehensive success metrics and KPIs based on the data analysis:

        Data Summary:
        {json.dumps(data_summary, indent=2)}

        Analysis Results:
        {json.dumps(analysis_results, indent=2)}

        Define metrics for:
        1. Data quality improvements
        2. Operational efficiency gains
        3. Business value creation
        4. Risk reduction achievements
        5. User adoption and satisfaction
        6. ROI and cost-benefit measures

        For each metric, provide:
        - Metric name and description
        - Current baseline (if determinable)
        - Target value and timeline
        - Measurement methodology
        - Reporting frequency
        - Responsible stakeholder

        Provide as JSON array of metric objects.
        """
        
        try:
            response = self.llm_handler.llm._call(prompt)
            try:
                metrics = json.loads(response)
                return metrics if isinstance(metrics, list) else [{"description": response, "source": "ai_generated"}]
            except json.JSONDecodeError:
                return [{"description": response, "source": "ai_generated"}]
        except Exception as e:
            logger.error(f"Error defining success metrics: {str(e)}")
            return self._fallback_success_metrics(data_summary)
    
    def _generate_risk_assessment_ai(self, data_summary: Dict[str, Any], 
                                    analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive risk assessment using AI"""
        
        prompt = f"""
        Conduct a comprehensive risk assessment for the data initiative:

        Data Context:
        {json.dumps(data_summary, indent=2)}

        Analysis Results:
        {json.dumps(analysis_results, indent=2)}

        Assess risks in:
        1. Data security and privacy
        2. Implementation and technical risks
        3. Business continuity risks
        4. Compliance and regulatory risks
        5. Financial and resource risks
        6. Change management risks

        For each risk category, provide:
        - Specific risk scenarios
        - Probability and impact assessment
        - Risk level (Critical/High/Medium/Low)
        - Mitigation strategies
        - Contingency plans
        - Monitoring recommendations

        Provide as structured JSON response.
        """
        
        try:
            response = self.llm_handler.llm._call(prompt)
            try:
                return json.loads(response)
            except json.JSONDecodeError:
                return {"risk_assessment": response, "source": "ai_generated"}
        except Exception as e:
            logger.error(f"Error generating risk assessment: {str(e)}")
            return {"error": "Could not generate risk assessment"}
    
    def _prepare_analysis_summary(self, analysis_results: Dict[str, Any], df: pd.DataFrame) -> str:
        """Use AI to prepare intelligent analysis summary"""
        
        data_summary = self._prepare_data_summary(df)
        
        prompt = f"""
        As a data analyst, create a concise but comprehensive analysis summary based on the following information:

        Dataset Information:
        {json.dumps(data_summary, indent=2)}

        Analysis Results:
        {json.dumps(analysis_results, indent=2)}

        Create a bullet-point summary that highlights:
        1. Key data characteristics and quality aspects
        2. Most important statistical findings
        3. Significant patterns or correlations discovered
        4. Data structure insights (numeric vs categorical features)
        5. Any notable anomalies or interesting discoveries

        Keep it concise but informative, focusing on business-relevant insights.
        Format as bullet points starting with '-'.
        """
        
        try:
            ai_summary = self.llm_handler.llm._call(prompt)
            return ai_summary
        except Exception as e:
            logger.error(f"Error generating AI analysis summary: {str(e)}")
            return f"Dataset with {len(df)} rows and {len(df.columns)} columns analyzed"
    
    def _fallback_data_quality_assessment(self, data_summary: Dict[str, Any]) -> Dict[str, Any]:
        """AI-powered fallback data quality assessment"""
        
        prompt = f"""
        As a data quality expert, analyze this dataset summary and provide a basic quality assessment:

        Data Summary:
        {json.dumps(data_summary, indent=2)}

        Provide a JSON response with:
        {{
            "critical_issues": [list of main data quality problems],
            "severity_level": "High/Medium/Low based on issues found",
            "recommendations": [list of 3-5 specific improvement recommendations],
            "business_impact": "brief description of how quality issues affect business",
            "source": "ai_fallback_analysis"
        }}

        Focus on practical, actionable insights based on the data characteristics.
        """
        
        try:
            response = self.llm_handler.llm._call(prompt)
            try:
                return json.loads(response)
            except json.JSONDecodeError:
                return {"assessment": response, "source": "ai_fallback_analysis"}
        except Exception as e:
            logger.error(f"Error in fallback data quality assessment: {str(e)}")
            return {"error": "Could not generate assessment", "source": "system_error"}
    
    def _fallback_risk_identification(self, data_summary: Dict[str, Any]) -> List[Dict[str, Any]]:
        """AI-powered fallback risk identification"""
        
        prompt = f"""
        As a business risk analyst, identify potential risks based on this dataset:

        Data Summary:
        {json.dumps(data_summary, indent=2)}

        Identify risks considering:
        - Data completeness and quality issues
        - Sample size limitations
        - Data distribution problems
        - Business continuity concerns

        Provide response as JSON array of risk objects:
        [
            {{
                "type": "risk category",
                "severity": "Critical/High/Medium/Low",
                "description": "detailed risk description",
                "business_impact": "how this affects business operations",
                "mitigation": "suggested mitigation approach",
                "source": "ai_fallback_analysis"
            }}
        ]
        """
        
        try:
            response = self.llm_handler.llm._call(prompt)
            try:
                risks = json.loads(response)
                return risks if isinstance(risks, list) else [{"description": response, "source": "ai_fallback_analysis"}]
            except json.JSONDecodeError:
                return [{"description": response, "source": "ai_fallback_analysis", "type": "AI Risk Analysis"}]
        except Exception as e:
            logger.error(f"Error in fallback risk identification: {str(e)}")
            return [{"error": "Could not identify risks", "source": "system_error"}]
    
    def _fallback_opportunity_identification(self, data_summary: Dict[str, Any]) -> List[Dict[str, Any]]:
        """AI-powered fallback opportunity identification"""
        
        prompt = f"""
        As a business strategist, identify opportunities based on this dataset analysis:

        Data Summary:
        {json.dumps(data_summary, indent=2)}

        Look for opportunities in:
        - Data utilization and analytics potential
        - Process improvements based on data insights
        - Business growth possibilities
        - Competitive advantages from data

        Provide response as JSON array:
        [
            {{
                "type": "opportunity category",
                "priority": "High/Medium/Low",
                "description": "detailed opportunity description",
                "potential_value": "expected business value",
                "implementation_effort": "Low/Medium/High",
                "timeline": "expected timeframe",
                "source": "ai_fallback_analysis"
            }}
        ]
        """
        
        try:
            response = self.llm_handler.llm._call(prompt)
            try:
                opportunities = json.loads(response)
                return opportunities if isinstance(opportunities, list) else [{"description": response, "source": "ai_fallback_analysis"}]
            except json.JSONDecodeError:
                return [{"description": response, "source": "ai_fallback_analysis", "type": "AI Opportunity Analysis"}]
        except Exception as e:
            logger.error(f"Error in fallback opportunity identification: {str(e)}")
            return [{"error": "Could not identify opportunities", "source": "system_error"}]
    
    def _fallback_action_prioritization(self, data_summary: Dict[str, Any]) -> List[Dict[str, Any]]:
        """AI-powered fallback action prioritization"""
        
        prompt = f"""
        As an operations consultant, prioritize actions based on this data analysis:

        Data Summary:
        {json.dumps(data_summary, indent=2)}

        Create a prioritized action plan considering:
        - Urgency and business impact
        - Implementation complexity and resources needed
        - Risk mitigation importance
        - Quick wins vs long-term investments

        Provide response as JSON array ordered by priority:
        [
            {{
                "action": "specific action name",
                "priority": "Critical/High/Medium/Low",
                "description": "detailed action description",
                "business_justification": "why this action is important",
                "effort": "Low/Medium/High implementation effort",
                "impact": "Low/Medium/High expected business impact",
                "timeline": "estimated completion time",
                "source": "ai_fallback_analysis"
            }}
        ]
        """
        
        try:
            response = self.llm_handler.llm._call(prompt)
            try:
                actions = json.loads(response)
                return actions if isinstance(actions, list) else [{"description": response, "source": "ai_fallback_analysis"}]
            except json.JSONDecodeError:
                return [{"description": response, "source": "ai_fallback_analysis", "type": "AI Action Prioritization"}]
        except Exception as e:
            logger.error(f"Error in fallback action prioritization: {str(e)}")
            return [{"error": "Could not prioritize actions", "source": "system_error"}]
    
    def _fallback_success_metrics(self, data_summary: Dict[str, Any]) -> List[Dict[str, Any]]:
        """AI-powered fallback success metrics definition"""
        
        prompt = f"""
        As a performance measurement expert, define success metrics based on this data analysis:

        Data Summary:
        {json.dumps(data_summary, indent=2)}

        Define comprehensive metrics covering:
        - Data quality improvements
        - Operational efficiency gains
        - Business value creation
        - Risk reduction achievements

        Provide response as JSON array:
        [
            {{
                "metric_name": "specific metric name",
                "description": "what this metric measures",
                "current_baseline": "current state if determinable from data",
                "target_value": "realistic target to achieve",
                "measurement_method": "how to measure this metric",
                "reporting_frequency": "how often to track",
                "business_relevance": "why this metric matters for business",
                "source": "ai_fallback_analysis"
            }}
        ]
        """
        
        try:
            response = self.llm_handler.llm._call(prompt)
            try:
                metrics = json.loads(response)
                return metrics if isinstance(metrics, list) else [{"description": response, "source": "ai_fallback_analysis"}]
            except json.JSONDecodeError:
                return [{"description": response, "source": "ai_fallback_analysis"}]
        except Exception as e:
            logger.error(f"Error in fallback success metrics: {str(e)}")
            return [{"error": "Could not define success metrics", "source": "system_error"}]
    
    def _generate_fallback_recommendations(self, analysis_results: Dict[str, Any], 
                                         df: pd.DataFrame) -> List[str]:
        """AI-powered fallback recommendations generation"""
        
        data_summary = self._prepare_data_summary(df)
        
        prompt = f"""
        As a business consultant, provide strategic recommendations based on this analysis:

        Dataset Summary:
        {json.dumps(data_summary, indent=2)}

        Analysis Results:
        {json.dumps(analysis_results, indent=2)}

        Provide 5-8 strategic recommendations that are:
        - Specific and actionable
        - Based on the actual data characteristics
        - Focused on business value creation
        - Considering implementation feasibility

        Return as a simple list of recommendation strings, each focusing on a different aspect:
        - Data quality and governance
        - Analytics and insights development
        - Process optimization
        - Risk management
        - Business growth opportunities
        """
        
        try:
            response = self.llm_handler.llm._call(prompt)
            recommendations = [line.strip().lstrip('- ').lstrip('* ').lstrip('1234567890. ') 
                             for line in response.split('\n') 
                             if line.strip() and not line.strip().startswith('#')]
            
            recommendations = [rec for rec in recommendations if len(rec) > 20]
            
            return recommendations if recommendations else [response]
        except Exception as e:
            logger.error(f"Error generating fallback recommendations: {str(e)}")
            return ["Error: Could not generate AI-powered recommendations"]