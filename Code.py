# import csv

# # Define the CSV file name
# filename = 'companies.csv'

# # Define the column names
# fields = ['company_name']

# # Define the data rows
# rows = [["Southwest",
# "Prudential",
# "Fannie Mae",
# "Raytheon Technologies (RTX)",
# "Vocalink A Mastercard company",
# "JPMorgan Chase",
# "CNA Insurance",
# "Takeda Pharmaceuticals U.S.A., Inc.",
# "Sumitomo Mitsui Banking Corporation- SMBC",
# "Ford"
# ]]

# # Write to CSV file
# with open(filename, mode='w', newline='') as csvfile:
#     csvwriter = csv.writer(csvfile)
#     csvwriter.writerow(fields)  # write the header
#     csvwriter.writerows(rows)   # write the data rows

# print(f"CSV file '{filename}' created successfully.")
# # import sys
# # sys.path.append(r'C:\Users\T Manoj\Desktop\bullet')
# # from news_scraper import NewsScraper
# # scraper = NewsScraper()
# # print(scraper.get_company_news("Apple"))


import requests
import json
import os
import re
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import time

@dataclass
class FinancialMetric:
    """Structured financial data"""
    value: float
    unit: str  # 'million', 'billion', 'dollars'
    period: str  # 'Q1 2025', 'FY 2024', etc.
    source: str
    confidence: float  # 0-1 scale

@dataclass
class EarningsData:
    """Structured earnings information"""
    company_name: str
    revenue: List[FinancialMetric]
    net_income: List[FinancialMetric]
    eps: List[FinancialMetric]
    guidance: List[str]
    key_metrics: Dict[str, List[FinancialMetric]]
    business_segments: Dict[str, str]
    challenges: List[str]
    growth_drivers: List[str]
    analyst_sentiment: str
    last_updated: datetime

class EnhancedEarningsResearcher:
    def _init_(self, tavily_api_key: Optional[str] = None):
        """
        Enhanced earnings researcher with improved accuracy
        
        Args:
            tavily_api_key: Your Tavily API key
        """
        self.tavily_api_key = tavily_api_key or os.getenv('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')
        self.tavily_url = "https://api.tavily.com/search"
        
        # Enhanced regex patterns for financial data
        self.financial_patterns = {
            'revenue': [
                r'(?:revenue|sales)\s*(?:of|was|reached|totaled)?\s*\$?(\d+(?:\.\d+)?)\s*(billion|million|B|M)\b',
                r'(?:net\s+)?revenue.?\$(\d+(?:\.\d+)?)\s(billion|million|B|M)',
                r'total\s+revenue.?\$(\d+(?:\.\d+)?)\s(billion|million|B|M)'
            ],
            'net_income': [
                r'(?:net\s+income|profit)\s*(?:of|was|reached)?\s*\$?(\d+(?:\.\d+)?)\s*(billion|million|B|M)',
                r'(?:quarterly|annual)\s+(?:net\s+)?(?:income|profit).?\$(\d+(?:\.\d+)?)\s(billion|million|B|M)'
            ],
            'eps': [
                r'(?:earnings\s+per\s+share|EPS)\s*(?:of|was|reached)?\s*\$?(\d+\.\d+)',
                r'diluted\s+EPS.*?\$(\d+\.\d+)',
                r'per\s+share.*?\$(\d+\.\d+)'
            ],
            'operating_income': [
                r'operating\s+income.?\$(\d+(?:\.\d+)?)\s(billion|million|B|M)'
            ],
            'free_cash_flow': [
                r'free\s+cash\s+flow.?\$(\d+(?:\.\d+)?)\s(billion|million|B|M)'
            ]
        }
        
        # Time period patterns
        self.period_patterns = [
            r'Q[1-4]\s+202[4-5]',
            r'(?:first|second|third|fourth)\s+quarter\s+202[4-5]',
            r'fiscal\s+(?:year\s+)?202[4-5]',
            r'FY\s*202[4-5]',
            r'full\s+year\s+202[4-5]'
        ]
        
    def search_with_validation(self, query: str, max_results: int = 10) -> List[Dict]:
        """Enhanced search with result validation"""
        if not self.tavily_api_key:
            raise ValueError("Tavily API key required for accurate data")
            
        payload = {
            "api_key": self.tavily_api_key,
            "query": query,
            "search_depth": "advanced",
            "include_answer": True,
            "include_raw_content": True,
            "max_results": max_results,
            "include_domains": [
                "sec.gov",
                "investor.*.com",
                "earnings.*.com",
                "bloomberg.com",
                "reuters.com",
                "cnbc.com",
                "marketwatch.com",
                "yahoo.com/finance"
            ]
        }
        
        try:
            response = requests.post(self.tavily_url, json=payload, timeout=30)
            response.raise_for_status()
            results = response.json().get('results', [])
            
            # Filter and score results by relevance and source quality
            scored_results = []
            for result in results:
                score = self._score_result_quality(result, query)
                result['quality_score'] = score
                if score > 0.3:  # Only include reasonably relevant results
                    scored_results.append(result)
            
            # Sort by quality score
            return sorted(scored_results, key=lambda x: x['quality_score'], reverse=True)
            
        except Exception as e:
            print(f"Search error: {e}")
            return []
    
    def _score_result_quality(self, result: Dict, query: str) -> float:
        """Score result quality based on source, recency, and relevance"""
        score = 0.5  # Base score
        
        url = result.get('url', '').lower()
        title = result.get('title', '').lower()
        content = result.get('content', '').lower()
        
        # Source quality scoring
        high_quality_domains = ['sec.gov', 'investor.', 'earnings.']
        medium_quality_domains = ['bloomberg.com', 'reuters.com', 'cnbc.com']
        
        for domain in high_quality_domains:
            if domain in url:
                score += 0.3
                break
        else:
            for domain in medium_quality_domains:
                if domain in url:
                    score += 0.2
                    break
        
        # Recency scoring (prefer recent results)
        current_year = datetime.now().year
        if f'{current_year}' in content or f'{current_year-1}' in content:
            score += 0.2
        
        # Relevance scoring
        query_terms = query.lower().split()
        for term in query_terms:
            if term in title:
                score += 0.1
            if term in content:
                score += 0.05
        
        return min(score, 1.0)  # Cap at 1.0
    
    def extract_financial_metrics_enhanced(self, search_results: List[Dict], company_name: str) -> Dict[str, List[FinancialMetric]]:
        """Enhanced financial metric extraction with validation"""
        
        extracted_metrics = {
            'revenue': [],
            'net_income': [],
            'eps': [],
            'operating_income': [],
            'free_cash_flow': []
        }
        
        for result in search_results:
            content = result.get('content', '') + ' ' + result.get('title', '')
            url = result.get('url', '')
            
            # Extract time periods first
            periods = re.findall('|'.join(self.period_patterns), content, re.IGNORECASE)
            
            for metric_type, patterns in self.financial_patterns.items():
                for pattern in patterns:
                    matches = re.finditer(pattern, content, re.IGNORECASE)
                    
                    for match in matches:
                        try:
                            if metric_type == 'eps':
                                value = float(match.group(1))
                                unit = 'dollars'
                            else:
                                value = float(match.group(1))
                                unit = match.group(2).lower()
                                if unit in ['b', 'billion']:
                                    unit = 'billion'
                                elif unit in ['m', 'million']:
                                    unit = 'million'
                            
                            # Find relevant time period
                            period = self._find_relevant_period(match.start(), content, periods)
                            
                            # Calculate confidence based on source quality and context
                            confidence = self._calculate_confidence(result, match, content)
                            
                            metric = FinancialMetric(
                                value=value,
                                unit=unit,
                                period=period,
                                source=url,
                                confidence=confidence
                            )
                            
                            # Avoid duplicates
                            if not self._is_duplicate_metric(metric, extracted_metrics[metric_type]):
                                extracted_metrics[metric_type].append(metric)
                                
                        except (ValueError, IndexError) as e:
                            continue
        
        # Sort by confidence and remove low-confidence duplicates
        for metric_type in extracted_metrics:
            extracted_metrics[metric_type] = self._clean_and_sort_metrics(
                extracted_metrics[metric_type]
            )
        
        return extracted_metrics
    
    def _find_relevant_period(self, match_position: int, content: str, periods: List[str]) -> str:
        """Find the most relevant time period for a financial metric"""
        if not periods:
            return "Unknown period"
        
        # Look for periods near the match
        content_around_match = content[max(0, match_position-200):match_position+200]
        
        for period in periods:
            if period.lower() in content_around_match.lower():
                return period
        
        # Return the most recent period found
        return max(periods, key=lambda p: '2025' in p or '2024' in p)
    
    def _calculate_confidence(self, result: Dict, match, content: str) -> float:
        """Calculate confidence score for a financial metric"""
        confidence = 0.5
        
        # Source quality
        url = result.get('url', '').lower()
        if 'sec.gov' in url or 'investor.' in url:
            confidence += 0.3
        elif any(domain in url for domain in ['bloomberg', 'reuters', 'cnbc']):
            confidence += 0.2
        
        # Context quality
        match_context = content[max(0, match.start()-100):match.start()+100].lower()
        
        # Positive indicators
        if any(word in match_context for word in ['reported', 'announced', 'generated', 'posted']):
            confidence += 0.1
        
        # Negative indicators (preliminary, estimated, etc.)
        if any(word in match_context for word in ['estimated', 'projected', 'preliminary', 'expected']):
            confidence -= 0.2
        
        return max(0.1, min(1.0, confidence))
    
    def _is_duplicate_metric(self, new_metric: FinancialMetric, existing_metrics: List[FinancialMetric]) -> bool:
        """Check if a metric is a duplicate"""
        for existing in existing_metrics:
            if (abs(new_metric.value - existing.value) < 0.01 and 
                new_metric.unit == existing.unit and
                new_metric.period == existing.period):
                return True
        return False
    
    def _clean_and_sort_metrics(self, metrics: List[FinancialMetric]) -> List[FinancialMetric]:
        """Clean and sort metrics by confidence"""
        # Remove very low confidence metrics
        cleaned = [m for m in metrics if m.confidence > 0.3]
        
        # Sort by confidence
        cleaned.sort(key=lambda x: x.confidence, reverse=True)
        
        # Keep only top 3 most confident metrics
        return cleaned[:3]
    
    def extract_qualitative_insights(self, search_results: List[Dict]) -> Dict[str, List[str]]:
        """Extract qualitative insights like guidance, challenges, growth drivers"""
        
        insights = {
            'guidance': [],
            'challenges': [],
            'growth_drivers': [],
            'business_segments': {}
        }
        
        guidance_keywords = ['guidance', 'outlook', 'forecast', 'expects', 'anticipates', 'projects']
        challenge_keywords = ['headwinds', 'challenges', 'risks', 'pressures', 'concerns', 'difficulties']
        growth_keywords = ['growth', 'expansion', 'investment', 'opportunity', 'initiative', 'driver']
        
        for result in search_results[:5]:  # Focus on top quality results
            content = result.get('content', '')
            
            # Extract guidance
            for keyword in guidance_keywords:
                pattern = f'.{{0,50}}{keyword}.{{0,200}}'
                matches = re.findall(pattern, content, re.IGNORECASE | re.DOTALL)
                for match in matches:
                    clean_match = re.sub(r'\s+', ' ', match.strip())
                    if len(clean_match) > 30 and clean_match not in insights['guidance']:
                        insights['guidance'].append(clean_match)
            
            # Extract challenges
            for keyword in challenge_keywords:
                pattern = f'.{{0,50}}{keyword}.{{0,200}}'
                matches = re.findall(pattern, content, re.IGNORECASE | re.DOTALL)
                for match in matches:
                    clean_match = re.sub(r'\s+', ' ', match.strip())
                    if len(clean_match) > 30 and clean_match not in insights['challenges']:
                        insights['challenges'].append(clean_match)
            
            # Extract growth drivers
            for keyword in growth_keywords:
                pattern = f'.{{0,50}}{keyword}.{{0,200}}'
                matches = re.findall(pattern, content, re.IGNORECASE | re.DOTALL)
                for match in matches:
                    clean_match = re.sub(r'\s+', ' ', match.strip())
                    if len(clean_match) > 30 and clean_match not in insights['growth_drivers']:
                        insights['growth_drivers'].append(clean_match)
        
        # Limit results
        for key in ['guidance', 'challenges', 'growth_drivers']:
            insights[key] = insights[key][:3]
        
        return insights
    
    def comprehensive_earnings_search(self, company_name: str) -> EarningsData:
        """Comprehensive earnings research with enhanced accuracy"""
        
        print(f":mag: Conducting comprehensive earnings research for {company_name}...")
        
        # Enhanced search queries
        search_queries = [
            f"{company_name} earnings results Q1 2025 revenue profit EPS",
            f"{company_name} quarterly financial results latest quarter SEC filing",
            f"{company_name} guidance forecast 2025 management outlook",
            f"{company_name} investor relations earnings call transcript",
            f"{company_name} business segments performance revenue breakdown",
            f"{company_name} challenges headwinds risks competitive",
            f'"{company_name}" 10-Q 10-K SEC filing financial statements'
        ]
        
        all_results = []
        
        for query in search_queries:
            print(f"  • Searching: {query[:50]}...")
            results = self.search_with_validation(query, max_results=8)
            all_results.extend(results)
            time.sleep(1)  # Rate limiting
        
        # Remove duplicates based on URL
        seen_urls = set()
        unique_results = []
        for result in all_results:
            url = result.get('url', '')
            if url not in seen_urls:
                seen_urls.add(url)
                unique_results.append(result)
        
        print(f"  • Found {len(unique_results)} unique sources")
        
        # Extract financial metrics
        financial_metrics = self.extract_financial_metrics_enhanced(unique_results, company_name)
        
        # Extract qualitative insights
        qualitative_insights = self.extract_qualitative_insights(unique_results)
        
        # Create structured earnings data
        earnings_data = EarningsData(
            company_name=company_name,
            revenue=financial_metrics.get('revenue', []),
            net_income=financial_metrics.get('net_income', []),
            eps=financial_metrics.get('eps', []),
            guidance=qualitative_insights.get('guidance', []),
            key_metrics={
                'operating_income': financial_metrics.get('operating_income', []),
                'free_cash_flow': financial_metrics.get('free_cash_flow', [])
            },
            business_segments=qualitative_insights.get('business_segments', {}),
            challenges=qualitative_insights.get('challenges', []),
            growth_drivers=qualitative_insights.get('growth_drivers', []),
            analyst_sentiment="",  # Could be enhanced with sentiment analysis
            last_updated=datetime.now()
        )
        
        return earnings_data
    
    def generate_enhanced_bulletin(self, earnings_data: EarningsData) -> str:
        """Generate enhanced bulletin with accuracy indicators"""
        
        def format_metric(metrics: List[FinancialMetric], metric_name: str) -> str:
            if not metrics:
                return f"{metric_name}: No reliable data found"
            
            result = f"{metric_name}:\n"
            for i, metric in enumerate(metrics[:3], 1):
                confidence_stars = "★" * int(metric.confidence * 5)
                if metric.unit in ['billion', 'million']:
                    result += f"    {i}. ${metric.value} {metric.unit} ({metric.period}) {confidence_stars}\n"
                else:
                    result += f"    {i}. ${metric.value} ({metric.period}) {confidence_stars}\n"
            return result
        
        bulletin = f"""
╔═══════════════════════════════════════════════════════════════════════════════════════╗
║                         ENHANCED EARNINGS BULLETIN                                    ║
║                              {earnings_data.company_name.upper()}                                        ║
║                         Generated: {earnings_data.last_updated.strftime('%Y-%m-%d %H:%M')}                            ║
╚═══════════════════════════════════════════════════════════════════════════════════════╝

:bar_chart: FINANCIAL PERFORMANCE (★ = Confidence Level)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

{format_metric(earnings_data.revenue, ":moneybag: REVENUE")}

{format_metric(earnings_data.net_income, ":chart_with_upwards_trend: NET INCOME")}

{format_metric(earnings_data.eps, ":dollar: EARNINGS PER SHARE")}

{format_metric(earnings_data.key_metrics.get('operating_income', []), ":gear:  OPERATING INCOME")}


:dart: MANAGEMENT GUIDANCE & OUTLOOK
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
        
        for i, guidance in enumerate(earnings_data.guidance[:3], 1):
            bulletin += f"{i}. {guidance}\n\n"
        
        if not earnings_data.guidance:
            bulletin += "No specific guidance information found in reliable sources.\n\n"
        
        bulletin += """:rocket: GROWTH DRIVERS & STRATEGIC INITIATIVES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
        
        for i, driver in enumerate(earnings_data.growth_drivers[:3], 1):
            bulletin += f"{i}. {driver}\n\n"
        
        if not earnings_data.growth_drivers:
            bulletin += "No specific growth driver information found in reliable sources.\n\n"
        
        bulletin += """:warning:  CHALLENGES & HEADWINDS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
        
        for i, challenge in enumerate(earnings_data.challenges[:3], 1):
            bulletin += f"{i}. {challenge}\n\n"
        
        if not earnings_data.challenges:
            bulletin += "No specific challenge information found in reliable sources.\n\n"
        
        bulletin += f"""



{'═' * 90}
Enhanced Earnings Research • {earnings_data.last_updated.strftime('%Y-%m-%d %H:%M:%S')}
{'═' * 90}
"""
        
        return bulletin
    
    def research_company_comprehensive(self, company_name: str, save_to_file: bool = True) -> str:
        """Main method for comprehensive earnings research"""
        
        try:
            # Comprehensive search and analysis
            earnings_data = self.comprehensive_earnings_search(company_name)
            
            # Generate enhanced bulletin
            bulletin = self.generate_enhanced_bulletin(earnings_data)
            
            # Save to file
            if save_to_file:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M')
                filename = f"{company_name.lower().replace(' ', '_')}_enhanced_earnings_{timestamp}.txt"
                
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write(bulletin)
                    
                    # Also save raw data as JSON
                    json_filename = filename.replace('.txt', '_data.json')
                    with open(json_filename, 'w', encoding='utf-8') as json_f:
                        # Convert earnings_data to dict for JSON serialization
                        data_dict = {
                            'company_name': earnings_data.company_name,
                            'revenue': [{'value': m.value, 'unit': m.unit, 'period': m.period, 'confidence': m.confidence} for m in earnings_data.revenue],
                            'net_income': [{'value': m.value, 'unit': m.unit, 'period': m.period, 'confidence': m.confidence} for m in earnings_data.net_income],
                            'eps': [{'value': m.value, 'unit': m.unit, 'period': m.period, 'confidence': m.confidence} for m in earnings_data.eps],
                            'guidance': earnings_data.guidance,
                            'challenges': earnings_data.challenges,
                            'growth_drivers': earnings_data.growth_drivers,
                            'last_updated': earnings_data.last_updated.isoformat()
                        }
                        json.dump(data_dict, json_f, indent=2)
                
                print(f":page_facing_up: Enhanced bulletin saved to: {filename}")
                print(f":bar_chart: Raw data saved to: {json_filename}")
            
            return bulletin
            
        except Exception as e:
            print(f":x: Research error: {e}")
            raise

# Usage Example
def main():
    """Enhanced usage example"""
    
    # Initialize with API key
    api_key = os.getenv('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') 
    
    researcher = EnhancedEarningsResearcher(tavily_api_key="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")
    
    
    mode = input("Choose mode:\n1. Search single company (interactive)\n2. Process CSV file\nEnter choice (1 or 2): ").strip()
    
    if mode == "1":
        while True:
            company = input("Enter company name: ").strip()
            print(f"\n:rocket: Starting enhanced earnings research for {company}...")
            bulletin = researcher.research_company_comprehensive(company)
            print("\n" + "="*50)
            print("RESEARCH COMPLETE")
            print("="*50)
            print(bulletin)
                
            continue_search = input("\n:arrows_counterclockwise: Search another company? (y/n): ").strip().lower()
            if continue_search != 'y':
                break
    
    elif mode == "2":
        csv_path = input(":file_folder: Enter CSV file path (or press Enter for default): ").strip()
        if not csv_path:
            csv_path = r"C:\Users\T Manoj\Desktop\bullet\data\companies.csv"  # Default path
            print(f"Using default path: {csv_path}")
            
        process_companies_from_csv(csv_path)
    

if _name_ == "_main_":
    main()
