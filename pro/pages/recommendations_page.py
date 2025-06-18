import dash_bootstrap_components as dbc
from dash import html, dcc, callback, Input, Output, State
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from database_config import load_data_from_database
from collections import Counter
import ast

def load_recommendations_data():
    """Load and return data for recommendations analysis"""
    return load_data_from_database()

def analyze_key_insights(df):
    """Analyze data to extract key insights for recommendations"""
    insights = {
        'total_reviews': len(df),
        'total_hotels': df['Hotel Name'].nunique() if 'Hotel Name' in df.columns else 0,
        'total_cities': df['City'].nunique() if 'City' in df.columns else 0,
        'avg_sentiment': 0,
        'top_complaints': [],
        'top_strengths': [],
        'worst_performing_cities': {},
        'best_performing_cities': {},
        'worst_performing_hotels': {},
        'best_performing_hotels': {},
        'critical_issues': [],
        'success_factors': [],
        'sentiment_distribution': {},
        'review_trends': {},
        'urgent_attention_needed': [],
        'improvement_opportunities': []
    }
    
    if df.empty:
        return insights
      # Convert sentiment to numeric for analysis
    df_numeric = convert_sentiment_to_numeric(df)
    
    # Basic metrics
    if 'sentiment_numeric' in df_numeric.columns:
        insights['avg_sentiment'] = df_numeric['sentiment_numeric'].mean()
        
        # Sentiment distribution analysis
        sentiment_bins = pd.cut(df_numeric['sentiment_numeric'], 
                               bins=[0, 20, 40, 60, 80, 100], 
                               labels=['Very Poor', 'Poor', 'Average', 'Good', 'Excellent'])
        insights['sentiment_distribution'] = sentiment_bins.value_counts().to_dict()
    
    # Time-based analysis if date column exists
    if 'Review Date' in df.columns:
        try:
            df_numeric['Review Date'] = pd.to_datetime(df_numeric['Review Date'], errors='coerce')
            monthly_trends = df_numeric.groupby(df_numeric['Review Date'].dt.to_period('M')).agg({
                'sentiment_numeric': 'mean'
            }).round(2)
            insights['review_trends'] = monthly_trends.to_dict()['sentiment_numeric'] if not monthly_trends.empty else {}
        except:
            pass
    
    # Analyze negative tokens (complaints)
    if 'negative_tokens' in df.columns:
        all_negative_tokens = []
        for tokens in df['negative_tokens'].dropna():
            if isinstance(tokens, str):
                try:
                    token_list = ast.literal_eval(tokens)
                    if isinstance(token_list, list):
                        all_negative_tokens.extend(token_list)
                except:
                    pass
        
        if all_negative_tokens:
            negative_counter = Counter(all_negative_tokens)
            insights['top_complaints'] = negative_counter.most_common(10)
    
    # Analyze positive tokens (strengths)
    if 'positive_tokens' in df.columns:
        all_positive_tokens = []
        for tokens in df['positive_tokens'].dropna():
            if isinstance(tokens, str):
                try:
                    token_list = ast.literal_eval(tokens)
                    if isinstance(token_list, list):
                        all_positive_tokens.extend(token_list)
                except:
                    pass
        
        if all_positive_tokens:
            positive_counter = Counter(all_positive_tokens)
            insights['top_strengths'] = positive_counter.most_common(10)    # City performance analysis with more detailed insights
    if 'City' in df.columns and 'sentiment_numeric' in df_numeric.columns:
        print(f"📊 Analyzing city performance for {df_numeric['City'].nunique()} cities...")
        
        # Check if sentiment data is actually varying
        sentiment_range = df_numeric['sentiment_numeric'].max() - df_numeric['sentiment_numeric'].min()
        sentiment_unique_count = df_numeric['sentiment_numeric'].nunique()
        print(f"📊 Sentiment range: {sentiment_range:.1f}, Unique values: {sentiment_unique_count}")
        
        if sentiment_range < 1.0 and sentiment_unique_count <= 3:
            print("⚠️ Warning: Sentiment data appears to have very low variance - all cities may show similar scores")
        
        city_performance = df_numeric.groupby('City').agg({
            'sentiment_numeric': ['mean', 'count', 'std', 'min', 'max'],
            'Hotel Name': 'nunique'
        }).round(2)
        city_performance.columns = ['avg_sentiment', 'review_count', 'sentiment_std', 'min_sentiment', 'max_sentiment', 'hotel_count']
        
        # Only calculate performance score if there's actual sentiment variation
        if sentiment_range > 5.0:  # Only if there's meaningful variation
            city_performance['performance_score'] = (
                city_performance['avg_sentiment'] * 
                np.log1p(city_performance['review_count']) / 
                (1 + city_performance['sentiment_std'].fillna(0))
            ).round(2)
        else:
            # If no variation, use review count and basic sentiment as score
            city_performance['performance_score'] = city_performance['avg_sentiment'].round(2)
        
        # Filter cities with meaningful data (at least 3 reviews for small datasets)
        min_reviews = max(3, min(5, len(df_numeric) // 20))  # Adaptive threshold
        reliable_cities = city_performance[city_performance['review_count'] >= min_reviews]
        
        print(f"📊 {len(reliable_cities)} cities have >= {min_reviews} reviews for analysis")
        
        if not reliable_cities.empty:
            # Sort by actual sentiment variation if available, otherwise by review count
            if sentiment_range > 5.0:
                worst_cities = reliable_cities.nsmallest(min(5, len(reliable_cities)), 'avg_sentiment')
                best_cities = reliable_cities.nlargest(min(5, len(reliable_cities)), 'avg_sentiment')
            else:
                # If no sentiment variation, show cities with most/least reviews
                worst_cities = reliable_cities.nsmallest(min(3, len(reliable_cities)), 'review_count')
                best_cities = reliable_cities.nlargest(min(3, len(reliable_cities)), 'review_count')
            
            insights['worst_performing_cities'] = {
                city: {
                    'avg_sentiment': row['avg_sentiment'],
                    'review_count': int(row['review_count']),
                    'hotel_count': int(row['hotel_count']),
                    'performance_score': row['performance_score'],
                    'sentiment_range': f"{row['min_sentiment']:.1f}-{row['max_sentiment']:.1f}"
                } for city, row in worst_cities.iterrows()
            }
            
            insights['best_performing_cities'] = {
                city: {
                    'avg_sentiment': row['avg_sentiment'],
                    'review_count': int(row['review_count']),
                    'hotel_count': int(row['hotel_count']),
                    'performance_score': row['performance_score'],
                    'sentiment_range': f"{row['min_sentiment']:.1f}-{row['max_sentiment']:.1f}"
                } for city, row in best_cities.iterrows()
            }
            
            # Only identify urgent cities if there's meaningful sentiment variation
            if sentiment_range > 10.0:
                urgent_cities = reliable_cities[
                    (reliable_cities['avg_sentiment'] < 40) & 
                    (reliable_cities['review_count'] >= min_reviews * 2)
                ]
                insights['urgent_attention_needed'] = list(urgent_cities.index)
            else:
                insights['urgent_attention_needed'] = []
                print("📊 No urgent cities identified due to low sentiment variance")
        else:
            print("⚠️ No cities have enough reviews for reliable analysis")
      # Hotel performance analysis with detailed metrics
    if 'Hotel Name' in df.columns and 'sentiment_numeric' in df_numeric.columns:
        hotel_performance = df_numeric.groupby('Hotel Name').agg({
            'sentiment_numeric': ['mean', 'count', 'std'],
            'City': 'first'  # Get the city for each hotel
        }).round(2)
        hotel_performance.columns = ['avg_sentiment', 'review_count', 'sentiment_std', 'city']
        
        # Calculate hotel performance score
        hotel_performance['performance_score'] = (
            hotel_performance['avg_sentiment'] * 
            np.log1p(hotel_performance['review_count']) / 
            (1 + hotel_performance['sentiment_std'].fillna(0))
        ).round(2)
        
        # Filter hotels with meaningful data (at least 10 reviews)
        reliable_hotels = hotel_performance[hotel_performance['review_count'] >= 10]
        
        if not reliable_hotels.empty:
            # Get actual data for worst and best performing hotels
            worst_hotels = reliable_hotels.nsmallest(min(5, len(reliable_hotels)), 'performance_score')
            best_hotels = reliable_hotels.nlargest(min(5, len(reliable_hotels)), 'performance_score')
            
            insights['worst_performing_hotels'] = {
                hotel: {
                    'avg_sentiment': row['avg_sentiment'],
                    'review_count': int(row['review_count']),
                    'city': row['city'],
                    'performance_score': row['performance_score']
                } for hotel, row in worst_hotels.iterrows()
            }
            
            insights['best_performing_hotels'] = {
                hotel: {
                    'avg_sentiment': row['avg_sentiment'],
                    'review_count': int(row['review_count']),
                    'city': row['city'],
                    'performance_score': row['performance_score']
                } for hotel, row in best_hotels.iterrows()
            }
    
    # Identify critical issues (sentiment < 30)
    if 'sentiment_numeric' in df_numeric.columns:
        critical_reviews = df_numeric[df_numeric['sentiment_numeric'] < 30]
        if not critical_reviews.empty and 'negative_tokens' in critical_reviews.columns:
            critical_tokens = []
            for tokens in critical_reviews['negative_tokens'].dropna():
                if isinstance(tokens, str):
                    try:
                        token_list = ast.literal_eval(tokens)
                        if isinstance(token_list, list):
                            critical_tokens.extend(token_list)
                    except:
                        pass
            
            if critical_tokens:
                critical_counter = Counter(critical_tokens)
                insights['critical_issues'] = critical_counter.most_common(5)
      # Identify success factors (sentiment > 80)
    if 'sentiment_numeric' in df_numeric.columns:
        excellent_reviews = df_numeric[df_numeric['sentiment_numeric'] > 80]
        if not excellent_reviews.empty and 'positive_tokens' in excellent_reviews.columns:
            success_tokens = []
            for tokens in excellent_reviews['positive_tokens'].dropna():
                if isinstance(tokens, str):
                    try:
                        token_list = ast.literal_eval(tokens)
                        if isinstance(token_list, list):
                            success_tokens.extend(token_list)
                    except:
                        pass
            
            if success_tokens:
                success_counter = Counter(success_tokens)
                insights['success_factors'] = success_counter.most_common(10)
    
    # Identify improvement opportunities based on moderate negative feedback
    if 'sentiment_numeric' in df_numeric.columns:
        moderate_negative = df_numeric[
            (df_numeric['sentiment_numeric'] >= 30) & 
            (df_numeric['sentiment_numeric'] < 60)
        ]
        if not moderate_negative.empty and 'negative_tokens' in moderate_negative.columns:
            improvement_tokens = []
            for tokens in moderate_negative['negative_tokens'].dropna():
                if isinstance(tokens, str):
                    try:
                        token_list = ast.literal_eval(tokens)
                        if isinstance(token_list, list):
                            improvement_tokens.extend(token_list)
                    except:
                        pass
            
            if improvement_tokens:
                improvement_counter = Counter(improvement_tokens)
                insights['improvement_opportunities'] = improvement_counter.most_common(8)
    
    return insights

def convert_sentiment_to_numeric(df):
    """Convert sentiment classification strings to numeric values (0-100 scale)"""
    if df.empty:
        return df
    
    df_copy = df.copy()
    
    # Check for different possible sentiment column names
    sentiment_columns = ['sentiment classification', 'sentiment_classification', 'sentiment', 'Sentiment', 'Sentiment Classification']
    sentiment_col = None
    
    for col in sentiment_columns:
        if col in df_copy.columns:
            sentiment_col = col
            break
    
    if sentiment_col is None:
        print("⚠️ No sentiment column found. Available columns:", list(df_copy.columns))
        return df_copy
    
    print(f"📊 Using sentiment column: '{sentiment_col}'")
    
    # Check unique values in sentiment column for debugging
    unique_sentiments = df_copy[sentiment_col].dropna().unique()
    print(f"📊 Unique sentiment values found: {unique_sentiments}")
    
    # Comprehensive sentiment mapping to handle various formats
    sentiment_mapping = {
        # Standard formats
        'Positive': 75.0, 'positive': 75.0, 'POSITIVE': 75.0,
        'Neutral': 50.0, 'neutral': 50.0, 'NEUTRAL': 50.0,
        'Negative': 25.0, 'negative': 25.0, 'NEGATIVE': 25.0,
        'Very Positive': 90.0, 'very positive': 90.0, 'VERY POSITIVE': 90.0,
        'Very Negative': 10.0, 'very negative': 10.0, 'VERY NEGATIVE': 10.0,
        'Excellent': 95.0, 'excellent': 95.0, 'EXCELLENT': 95.0,
        'Poor': 5.0, 'poor': 5.0, 'POOR': 5.0,
        
        # Additional possible formats
        'Good': 70.0, 'good': 70.0, 'GOOD': 70.0,
        'Bad': 20.0, 'bad': 20.0, 'BAD': 20.0,
        'Average': 50.0, 'average': 50.0, 'AVERAGE': 50.0,
        'Fair': 45.0, 'fair': 45.0, 'FAIR': 45.0,
        'Great': 85.0, 'great': 85.0, 'GREAT': 85.0,
        'Terrible': 5.0, 'terrible': 5.0, 'TERRIBLE': 5.0,
        'Amazing': 95.0, 'amazing': 95.0, 'AMAZING': 95.0,
        
        # Numeric strings (in case sentiments are stored as strings)
        '1': 10.0, '2': 25.0, '3': 50.0, '4': 75.0, '5': 90.0,
        '1.0': 10.0, '2.0': 25.0, '3.0': 50.0, '4.0': 75.0, '5.0': 90.0
    }
    
    # Clean the sentiment values (remove extra whitespace)
    df_copy[sentiment_col] = df_copy[sentiment_col].astype(str).str.strip()
    
    # Map sentiments to numeric values
    df_copy['sentiment_numeric'] = df_copy[sentiment_col].map(sentiment_mapping)
    
    # Handle unmapped values more intelligently
    unmapped_count = df_copy['sentiment_numeric'].isna().sum()
    if unmapped_count > 0:
        print(f"⚠️ {unmapped_count} sentiment values could not be mapped:")
        unmapped_values = df_copy[df_copy['sentiment_numeric'].isna()][sentiment_col].unique()
        print(f"Unmapped values: {unmapped_values}")
        
        # Try to handle unmapped values intelligently
        for idx, row in df_copy[df_copy['sentiment_numeric'].isna()].iterrows():
            sentiment_val = str(row[sentiment_col]).lower().strip()
            
            # Try to parse as numeric score
            try:
                numeric_val = float(sentiment_val)
                if 0 <= numeric_val <= 1:  # Probability score
                    df_copy.loc[idx, 'sentiment_numeric'] = numeric_val * 100
                elif 1 < numeric_val <= 5:  # 1-5 scale
                    df_copy.loc[idx, 'sentiment_numeric'] = (numeric_val - 1) * 25 + 10
                elif 0 <= numeric_val <= 100:  # Already 0-100 scale
                    df_copy.loc[idx, 'sentiment_numeric'] = numeric_val
                else:
                    df_copy.loc[idx, 'sentiment_numeric'] = 50.0  # Default
            except:
                # Check for partial matches
                if 'pos' in sentiment_val:
                    df_copy.loc[idx, 'sentiment_numeric'] = 75.0
                elif 'neg' in sentiment_val:
                    df_copy.loc[idx, 'sentiment_numeric'] = 25.0
                elif 'neu' in sentiment_val:
                    df_copy.loc[idx, 'sentiment_numeric'] = 50.0
                else:
                    df_copy.loc[idx, 'sentiment_numeric'] = 50.0  # Default
    
    # Final check
    final_unmapped = df_copy['sentiment_numeric'].isna().sum()
    if final_unmapped > 0:
        print(f"⚠️ Still {final_unmapped} unmapped values, filling with 50.0")
        df_copy['sentiment_numeric'] = df_copy['sentiment_numeric'].fillna(50.0)
    
    # Show the distribution of converted sentiments
    sentiment_dist = df_copy['sentiment_numeric'].value_counts().sort_index()
    print(f"📊 Sentiment distribution after conversion:\n{sentiment_dist}")
    
    return df_copy

def generate_recommendations(insights):
    """Generate actionable recommendations based on insights"""
    recommendations = {
        'immediate_actions': [],
        'short_term_strategies': [],
        'long_term_initiatives': [],
        'monitoring_systems': []
    }
    
    # Immediate Actions based on critical issues and data
    if insights['critical_issues']:
        for issue, count in insights['critical_issues']:
            if count >= 3:  # Adjust threshold based on actual data
                recommendations['immediate_actions'].append({
                    'issue': issue.title(),
                    'count': count,
                    'action': get_dynamic_action_for_issue(issue, count, insights),
                    'priority': 'CRITICAL' if count > 10 else 'HIGH',
                    'affected_reviews': f"{count} reviews mention this issue"
                })
    
    # Add urgent city attention to immediate actions
    if insights['urgent_attention_needed']:
        for city in insights['urgent_attention_needed']:
            city_data = insights['worst_performing_cities'].get(city, {})
            recommendations['immediate_actions'].append({
                'issue': f"Poor Performance in {city}",
                'count': city_data.get('review_count', 0),
                'action': f"Deploy emergency quality improvement team to {city}. Focus on hotels with lowest ratings.",
                'priority': 'URGENT',
                'affected_reviews': f"{city_data.get('review_count', 0)} reviews, avg sentiment: {city_data.get('avg_sentiment', 0):.1f}"
            })
    
    # Short-term strategies based on improvement opportunities
    if insights['improvement_opportunities']:
        for opportunity, count in insights['improvement_opportunities']:
            if count >= 5:
                recommendations['short_term_strategies'].append({
                    'opportunity': opportunity.title(),
                    'count': count,
                    'strategy': get_dynamic_strategy_for_issue(opportunity, count, insights),
                    'timeline': '2-4 months',
                    'impact': f"Could improve {count} customer experiences"
                })
    
    # Add city-specific strategies
    if insights['worst_performing_cities']:
        for city, data in list(insights['worst_performing_cities'].items())[:3]:  # Top 3 worst
            recommendations['short_term_strategies'].append({
                'opportunity': f"Service Enhancement in {city}",
                'count': data['review_count'],
                'strategy': f"Implement comprehensive service training for all {data['hotel_count']} hotels in {city}. Focus on staff behavior and facility maintenance.",
                'timeline': '3-6 months',
                'impact': f"Potential to improve {data['review_count']} customer experiences"
            })
    
    # Long-term initiatives based on success factors and best practices
    if insights['success_factors']:
        for strength, count in insights['success_factors'][:5]:  # Top 5 strengths
            recommendations['long_term_initiatives'].append({
                'strength': strength.title(),
                'count': count,
                'initiative': get_dynamic_initiative_for_strength(strength, count, insights),
                'timeline': '6-18 months',
                'scalability': f"Replicate across all hotels - {count} customers value this"
            })
    
    # Add best practice replication initiatives
    if insights['best_performing_cities']:
        top_city = next(iter(insights['best_performing_cities'].items()))
        city_name, city_data = top_city
        recommendations['long_term_initiatives'].append({
            'strength': f"Best Practices from {city_name}",
            'count': city_data['review_count'],
            'initiative': f"Study and document successful practices from {city_name} (avg sentiment: {city_data['avg_sentiment']:.1f}). Create standardized protocols for other cities.",
            'timeline': '12-24 months',
            'scalability': f"Model for {len(insights['worst_performing_cities'])} underperforming cities"
        })
    
    # Dynamic monitoring systems based on actual data patterns
    monitoring_systems = []
    
    if insights['total_reviews'] > 100:
        monitoring_systems.append({
            'system': 'Real-time Sentiment Monitoring',
            'description': f'Track sentiment across {insights["total_cities"]} cities and {insights["total_hotels"]} hotels',
            'frequency': 'Daily',
            'kpis': [f'Target: Maintain >70% positive sentiment (current: {insights["avg_sentiment"]:.1f}%)', 'Weekly trend analysis', 'City performance ranking']
        })
    
    if insights['critical_issues']:
        monitoring_systems.append({
            'system': 'Issue Tracking Dashboard',
            'description': f'Monitor top {len(insights["critical_issues"])} critical issues in real-time',
            'frequency': 'Weekly',
            'kpis': [f'Track mentions of: {", ".join([issue for issue, _ in insights["critical_issues"][:3]])}', 'Resolution time tracking', 'Improvement trend analysis']
        })
    
    if insights['sentiment_distribution']:
        poor_reviews = insights['sentiment_distribution'].get('Very Poor', 0) + insights['sentiment_distribution'].get('Poor', 0)
        if poor_reviews > 0:
            monitoring_systems.append({
                'system': 'Quality Assurance Program',
                'description': f'Address {poor_reviews} poor-quality experiences through systematic monitoring',                'frequency': 'Monthly',
                'kpis': ['Reduce poor reviews by 50%', 'Increase excellent reviews', 'Customer satisfaction improvement']
            })
    
    recommendations['monitoring_systems'] = monitoring_systems
    
    return recommendations

def get_dynamic_action_for_issue(issue, count, insights):
    """Get dynamic action based on actual data context"""
    severity = "immediate" if count > 15 else "urgent" if count > 8 else "priority"
    total_reviews = insights.get('total_reviews', 1)
    percentage = (count / total_reviews) * 100
    
    action_templates = {
        'dirty': f'Launch {severity} deep-cleaning initiative. {percentage:.1f}% of reviews mention cleanliness issues.',
        'rude': f'Implement {severity} customer service retraining. {count} complaints about staff behavior.',
        'broken': f'Deploy {severity} maintenance teams. Equipment failures reported in {count} reviews.',
        'slow': f'Optimize service processes {severity}ly. {count} customers report delays.',
        'expensive': f'Review pricing strategy {severity}ly. {percentage:.1f}% find services overpriced.',
        'noisy': f'Install soundproofing {severity}ly. Noise complaints from {count} guests.',
        'cold': f'Upgrade heating systems {severity}ly. Temperature issues in {count} reviews.',
        'small': f'Optimize space utilization {severity}ly. {count} guests report cramped conditions.',
        'poor': f'Comprehensive quality review needed {severity}ly. {count} poor service reports.',
        'bad': f'Emergency service improvement required. {percentage:.1f}% report poor experiences.'
    }
    
    return action_templates.get(issue.lower(), f'Address {issue} issues {severity}ly - reported by {count} customers ({percentage:.1f}% of reviews)')

def get_dynamic_strategy_for_issue(issue, count, insights):
    """Get dynamic strategy based on actual data context"""
    total_hotels = insights.get('total_hotels', 1)
    avg_sentiment = insights.get('avg_sentiment', 50)
    
    strategy_templates = {
        'service': f'Develop comprehensive staff training across {total_hotels} hotels. Current sentiment: {avg_sentiment:.1f}/100',
        'food': f'Review menu and kitchen operations in all locations. {count} customers report food issues.',
        'room': f'Upgrade room facilities systematically across {total_hotels} properties.',
        'staff': f'Implement staff development program. {count} reports of staff-related issues.',
        'cleanliness': f'Establish rigorous cleaning standards across all {total_hotels} hotels.',
        'location': f'Improve location amenities and accessibility information for guests.',
        'price': f'Reassess value proposition across all properties. {count} price-related complaints.',
        'wifi': f'Upgrade internet infrastructure in all {total_hotels} locations.',
        'breakfast': f'Enhance breakfast offerings and service quality across properties.',
        'parking': f'Improve parking facilities and information for guests.'
    }
    
    return strategy_templates.get(issue.lower(), f'Develop targeted improvement strategy for {issue} - affecting {count} customers')

def get_dynamic_initiative_for_strength(strength, count, insights):
    """Get dynamic initiative based on actual success factors"""
    total_hotels = insights.get('total_hotels', 1)
    best_cities = list(insights.get('best_performing_cities', {}).keys())[:2]
    
    initiative_templates = {
        'helpful': f'Replicate helpful staff practices from top-performing cities across {total_hotels} hotels.',
        'clean': f'Standardize excellent cleaning protocols across all properties. {count} customers praise this.',
        'friendly': f'Scale friendly service training to all locations based on {count} positive mentions.',
        'comfortable': f'Expand comfort features that {count} customers appreciate to all properties.',
        'convenient': f'Implement convenience features across all {total_hotels} hotels.',
        'beautiful': f'Enhance aesthetic appeal based on what {count} customers love.',
        'quiet': f'Replicate peaceful environment strategies to all locations.',
        'spacious': f'Optimize space design across properties based on {count} positive feedback.',
        'excellent': f'Document and replicate excellent service standards from best locations.',
        'amazing': f'Identify and scale amazing experience elements that {count} customers mention.'
    }    
    cities_context = f" Focus on replicating best practices from {', '.join(best_cities)}" if best_cities else ""
    
    return initiative_templates.get(strength.lower(), f'Scale {strength} practices across all properties{cities_context} - {count} customers value this')

def get_action_for_issue(issue):
    """Get specific action for critical issues"""
    action_map = {
        'dirty': 'Implement enhanced cleaning protocols and regular inspections',
        'rude': 'Conduct immediate customer service training for all staff',
        'broken': 'Establish urgent maintenance response team and equipment audit',
        'slow': 'Review and optimize service processes, add staff during peak times',
        'expensive': 'Review pricing strategy and implement value-added packages',
        'noisy': 'Install soundproofing and implement quiet hours policies',
        'cold': 'Upgrade heating systems and provide additional amenities',
        'small': 'Optimize room layouts and enhance storage solutions',
        'poor': 'Conduct comprehensive service quality assessment and improvements',
        'bad': 'Implement immediate quality control measures across all services'
    }
    return action_map.get(issue.lower(), f'Address {issue} issues through targeted improvement programs')

def get_strategy_for_complaint(complaint):
    """Get strategy for addressing complaints"""
    strategy_map = {
        'service': 'Develop comprehensive staff training program focusing on customer service excellence',
        'food': 'Review menu offerings, improve food quality, and enhance dining experience',
        'room': 'Upgrade room facilities, improve maintenance schedules, and enhance amenities',
        'staff': 'Implement staff development programs and improve communication training',
        'cleanliness': 'Establish rigorous cleaning standards and regular quality audits',
        'location': 'Improve transportation options and provide better area information',
        'breakfast': 'Enhance breakfast offerings with local and international options',
        'wifi': 'Upgrade internet infrastructure and provide reliable connectivity',
        'pool': 'Improve pool facilities, maintenance, and surrounding amenities',
        'parking': 'Expand parking options or provide valet services'
    }
    return strategy_map.get(complaint.lower(), f'Develop targeted strategy to address {complaint} concerns')

def get_initiative_for_strength(strength):
    """Get initiative for leveraging strengths"""
    initiative_map = {
        'helpful': 'Create staff recognition program and share best practices across all properties',
        'clean': 'Develop cleanliness as a brand differentiator and marketing advantage',
        'comfortable': 'Invest in comfort-enhancing amenities and promote comfort guarantee',
        'friendly': 'Build hospitality culture program and train staff as local ambassadors',
        'good': 'Establish quality excellence standards and continuous improvement processes',
        'nice': 'Develop customer experience programs that enhance overall satisfaction',
        'great': 'Create premium service tiers and loyalty programs for exceptional experiences',
        'excellent': 'Establish centers of excellence and best practice sharing programs',
        'beautiful': 'Invest in aesthetic improvements and promote visual appeal in marketing',
        'convenient': 'Enhance convenience features and promote accessibility advantages'
    }
    return initiative_map.get(strength.lower(), f'Leverage {strength} as a competitive advantage through strategic marketing and enhancement')

def create_recommendations_content(city_filter="all", start_date=None, end_date=None):
    """Create the main recommendations page content"""
    try:
        print("📋 Loading Recommendations page content...")
        df = load_recommendations_data()
        
        if df.empty:
            return html.Div([
                html.H2("📋 Government Recommendations", className="page-title"),
                html.P("No data available for generating recommendations", className="text-muted")
            ])
        
        # Filter data based on selections
        filtered_df = df.copy()
        if city_filter and city_filter != "all":
            filtered_df = filtered_df[filtered_df['City'] == city_filter] if 'City' in filtered_df.columns else filtered_df
        
        if start_date and end_date:
            if 'Review Date' in filtered_df.columns:
                filtered_df['Review Date'] = pd.to_datetime(filtered_df['Review Date'], errors='coerce')
                filtered_df = filtered_df[
                    (filtered_df['Review Date'] >= start_date) & 
                    (filtered_df['Review Date'] <= end_date)
                ]
        
        # Analyze insights and generate recommendations
        insights = analyze_key_insights(filtered_df)
        recommendations = generate_recommendations(insights)
        
        return html.Div([
            create_recommendations_header(),
            create_executive_summary_section(insights),
            create_key_insights_section(insights),
            create_immediate_actions_section(recommendations['immediate_actions']),
            create_strategic_recommendations_section(recommendations),
            create_monitoring_framework_section(recommendations['monitoring_systems']),
            create_implementation_roadmap_section(recommendations)
        ], className="recommendations-content")
        
    except Exception as e:
        print(f"Error creating recommendations content: {e}")
        return html.Div([
            html.H2("📋 Government Recommendations", className="page-title"),
            html.P(f"Error loading recommendations: {str(e)}", className="text-danger")
        ])

def create_recommendations_header():
    """Create the recommendations page header"""
    return html.Div([
        html.Div([
            html.H1([
                html.I(className="fas fa-lightbulb me-3"),
                "Government Tourism Recommendations"
            ], className="enhanced-title text-white"),
            html.P([
                "📊 Data-driven actionable insights for tourism industry improvement and strategic policy development"
            ], className="enhanced-subtitle text-white-50")
        ])
    ], className="recommendations-header")

def create_executive_summary_section(insights):
    """Create executive summary with key metrics"""
    sentiment_status = "Excellent" if insights['avg_sentiment'] >= 80 else \
                      "Good" if insights['avg_sentiment'] >= 65 else \
                      "Average" if insights['avg_sentiment'] >= 45 else \
                      "Needs Improvement"
    
    sentiment_color = "#10b981" if insights['avg_sentiment'] >= 80 else \
                     "#3b82f6" if insights['avg_sentiment'] >= 65 else \
                     "#f59e0b" if insights['avg_sentiment'] >= 45 else \
                     "#ef4444"
    
    return html.Div([
        html.H2([
            html.I(className="fas fa-chart-pie me-2"),
            "Executive Summary"
        ], className="section-title"),
        
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H4(f"{insights['total_reviews']:,}", className="metric-value text-primary"),
                        html.P("Total Reviews Analyzed", className="metric-label"),
                        html.Small("Comprehensive dataset", className="text-muted")
                    ])
                ], className="metric-card h-100")
            ], md=3),
            
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H4(f"{insights['avg_sentiment']:.1f}/100", className="metric-value", style={"color": sentiment_color}),
                        html.P("Overall Sentiment Score", className="metric-label"),
                        html.Small(f"Status: {sentiment_status}", className="text-muted")
                    ])
                ], className="metric-card h-100")
            ], md=3),
            
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H4(f"{insights['total_cities']}", className="metric-value text-info"),
                        html.P("Cities Covered", className="metric-label"),
                        html.Small("Geographic coverage", className="text-muted")
                    ])
                ], className="metric-card h-100")
            ], md=3),
            
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H4(f"{insights['total_hotels']}", className="metric-value text-success"),
                        html.P("Hotels Analyzed", className="metric-label"),
                        html.Small("Industry scope", className="text-muted")
                    ])
                ], className="metric-card h-100")
            ], md=3)
        ], className="mb-4")
    ], className="executive-summary-section mb-4")

def create_key_insights_section(insights):
    """Create key insights section with data visualizations"""
    return html.Div([
        html.H2([
            html.I(className="fas fa-search me-2"),
            "Key Insights & Findings"
        ], className="section-title"),
        
        dbc.Row([
            dbc.Col([
                html.Div([
                    html.H4("🔴 Top Areas for Improvement", className="chart-title"),
                    html.Div(id="top-complaints-chart")
                ])
            ], lg=6),
            
            dbc.Col([
                html.Div([
                    html.H4("🟢 Tourism Strengths", className="chart-title"),
                    html.Div(id="top-strengths-chart")
                ])
            ], lg=6)
        ], className="mb-4"),
        
        dbc.Row([
            dbc.Col([
                html.Div([
                    html.H4("📍 City Performance Analysis", className="chart-title"),
                    create_city_performance_insights(insights)
                ])
            ], lg=12)
        ])
    ], className="key-insights-section mb-4")

def create_city_performance_insights(insights):
    """Create city performance insights with better diagnostics"""
    if not insights['worst_performing_cities'] and not insights['best_performing_cities']:
        return html.Div([
            html.P("Insufficient data for city performance analysis", className="text-muted"),
            html.Small("Need at least 3 reviews per city for meaningful analysis", className="text-info")
        ])
    
    content = []
    
    # Add diagnostic information about sentiment data quality
    all_cities_data = {**insights.get('worst_performing_cities', {}), **insights.get('best_performing_cities', {})}
    if all_cities_data:
        sentiment_values = [data['avg_sentiment'] for data in all_cities_data.values()]
        sentiment_range = max(sentiment_values) - min(sentiment_values)
        
        if sentiment_range < 5.0:
            content.append(html.Div([
                dbc.Alert([
                    html.H6("📊 Data Quality Notice", className="alert-heading"),
                    html.P(f"Limited sentiment variation detected (range: {sentiment_range:.1f}). This may indicate:"),
                    html.Ul([
                        html.Li("Sentiment analysis needs calibration"),
                        html.Li("Data preprocessing issues"),
                        html.Li("Homogeneous review quality across cities")
                    ]),
                    html.P("Rankings below are based on review volume and available data.", className="mb-0")
                ], color="info", className="mb-3")
            ]))
    
    if insights['worst_performing_cities']:
        city_items = []
        for city, data in list(insights['worst_performing_cities'].items())[:3]:
            sentiment_range = data.get('sentiment_range', 'N/A')
            city_items.append(html.Div([
                html.Strong(city),
                html.Span(f" - {data['avg_sentiment']:.1f}/100", className="text-muted ms-2"),
                html.Span(f" ({data['review_count']} reviews, {data['hotel_count']} hotels)", className="text-muted ms-2"),
                html.Br(),
                html.Small(f"Sentiment range: {sentiment_range}", className="text-info me-3"),
                html.Small(get_enhanced_city_recommendation(data), className="text-warning")
            ], className="mb-2"))
        
        content.append(html.Div([
            html.H5("⚠️ Cities Requiring Attention", className="text-warning mb-3"),
            html.Div(city_items)
        ], className="mb-4"))
    
    if insights['best_performing_cities']:
        city_items = []
        for city, data in list(insights['best_performing_cities'].items())[:3]:
            sentiment_range = data.get('sentiment_range', 'N/A')
            city_items.append(html.Div([
                html.Strong(city),
                html.Span(f" - {data['avg_sentiment']:.1f}/100", className="text-muted ms-2"),
                html.Span(f" ({data['review_count']} reviews, {data['hotel_count']} hotels)", className="text-muted ms-2"),
                html.Br(),
                html.Small(f"Sentiment range: {sentiment_range}", className="text-info me-3"),
                html.Small(get_enhanced_city_success_factor(data), className="text-success")
            ], className="mb-2"))
        
        content.append(html.Div([
            html.H5("🌟 High-Activity Cities", className="text-success mb-3"),
            html.Div(city_items)
        ]))
    
    return html.Div(content)

def get_enhanced_city_recommendation(data):
    """Get enhanced recommendation based on city data"""
    sentiment_score = data['avg_sentiment']
    review_count = data['review_count']
    
    if abs(sentiment_score - 50.0) < 1.0:  # Very close to default value
        return f"Review data quality - {review_count} reviews available for analysis"
    elif sentiment_score < 40:
        return f"Critical: Comprehensive tourism review needed ({review_count} reviews)"
    elif sentiment_score < 55:
        return f"Priority: Focus on service improvements ({review_count} reviews)"
    else:
        return f"Monitor: Enhance specific areas identified ({review_count} reviews)"

def get_enhanced_city_success_factor(data):
    """Get enhanced success factor description for cities"""
    sentiment_score = data['avg_sentiment']
    review_count = data['review_count']
    
    if abs(sentiment_score - 50.0) < 1.0:  # Very close to default value
        return f"High review volume - {review_count} reviews for pattern analysis"
    elif sentiment_score >= 85:
        return f"Excellence: Model for other destinations ({review_count} reviews)"
    elif sentiment_score >= 75:
        return f"Strong performance: Maintain standards ({review_count} reviews)"
    else:
        return f"Good foundation: Build on strengths ({review_count} reviews)"

def get_city_recommendation(sentiment_score):
    """Get recommendation based on city sentiment score"""
    if sentiment_score < 40:
        return "Critical: Requires comprehensive tourism infrastructure review and immediate intervention"
    elif sentiment_score < 55:
        return "Priority: Focus on hospitality training and service quality improvements"
    else:
        return "Moderate: Enhance specific areas identified in negative feedback"

def get_city_success_factor(sentiment_score):
    """Get success factor description for high-performing cities"""
    if sentiment_score >= 85:
        return "Excellence: Use as best practice model for other destinations"
    elif sentiment_score >= 75:
        return "Strong performance: Maintain standards and expand successful practices"
    else:
        return "Good foundation: Build on existing strengths for further improvement"

def create_immediate_actions_section(immediate_actions):
    """Create immediate actions section with dynamic content"""
    if not immediate_actions:
        return html.Div([
            html.H2([
                html.I(className="fas fa-check-circle me-2 text-success"),
                "No Critical Issues Identified"
            ], className="section-title"),
            dbc.Alert([
                html.H5("Good News!", className="alert-heading"),
                html.P("Based on current data analysis, no critical issues requiring immediate action have been identified."),
                html.P("Continue regular monitoring and focus on strategic improvements.", className="mb-0")
            ], color="success")
        ], className="immediate-actions-section mb-4")
    
    return html.Div([
        html.H2([
            html.I(className="fas fa-exclamation-triangle me-2 text-warning"),
            "Immediate Actions Required"
        ], className="section-title"),
        
        html.Div([
            dbc.Alert([
                html.H5([
                    html.I(className="fas fa-fire me-2"),
                    f"Critical Issues Identified ({len(immediate_actions)} items)"
                ], className="alert-heading"),
                html.P("These issues require immediate government attention and intervention:", className="mb-3"),
                
                html.Div([
                    dbc.Card([
                        dbc.CardBody([
                            html.Div([                                html.H6([
                                    html.Span("🚨", className="me-2"),
                                    action['issue'],
                                    dbc.Badge(
                                        action['priority'], 
                                        color="danger" if action['priority'] in ['CRITICAL', 'URGENT'] else "warning", 
                                        className="ms-2"
                                    )
                                ], className="fw-bold"),
                                html.P(action['action'], className="mb-2"),
                                html.Div([
                                    html.Small([
                                        html.I(className="fas fa-chart-bar me-1"),
                                        action.get('affected_reviews', f"{action['count']} mentions")
                                    ], className="text-muted me-3"),
                                    html.Small([
                                        html.Strong("Priority Level: "),
                                        html.Span(action['priority'], className="text-danger fw-bold")
                                    ])
                                ])
                            ])
                        ])
                    ], className="mb-3") for action in immediate_actions
                ])
            ], color="warning", className="border-start border-warning border-4")
        ])
    ], className="immediate-actions-section mb-4")

def create_strategic_recommendations_section(recommendations):
    """Create strategic recommendations section"""
    return html.Div([
        html.H2([
            html.I(className="fas fa-chess me-2"),
            "Strategic Recommendations"
        ], className="section-title"),
        
        dbc.Row([
            dbc.Col([
                create_strategy_card(
                    "🎯 Short-Term Strategies",
                    "3-6 Month Implementation",
                    recommendations['short_term_strategies'],
                    "primary"
                )
            ], lg=6),
            
            dbc.Col([
                create_strategy_card(
                    "🚀 Long-Term Initiatives",
                    "1-2 Year Implementation",
                    recommendations['long_term_initiatives'],
                    "success"
                )
            ], lg=6)
        ])
    ], className="strategic-recommendations-section mb-4")

def create_strategy_card(title, timeline, strategies, color):
    """Create a strategy card with dynamic content"""
    if not strategies:
        return dbc.Card([
            dbc.CardHeader([
                html.H4(title, className="mb-0"),
                html.Small(timeline, className="text-muted")
            ]),
            dbc.CardBody([
                html.P("No specific strategies needed based on current data", className="text-muted"),
                html.Small("Continue monitoring for emerging patterns")
            ])
        ], color=color, outline=True, className="h-100")
    
    strategy_items = []
    for strategy in strategies[:4]:  # Show top 4 strategies
        # Handle both short-term strategies and long-term initiatives
        if 'opportunity' in strategy:
            # Short-term strategy
            strategy_items.append(html.Div([
                html.H6([
                    html.I(className="fas fa-target me-2"),
                    strategy['opportunity']
                ], className="fw-bold text-primary"),
                html.P(strategy['strategy'], className="mb-2"),                html.Div([
                    dbc.Badge(f"Affects {strategy['count']} reviews", color="light", className="me-2"),
                    html.Small([
                        html.Strong("Timeline: "),
                        strategy['timeline']
                    ], className="text-muted")
                ], className="d-flex justify-content-between align-items-center"),
                html.Small(strategy['impact'], className="text-info")
            ], className="mb-3"))
        elif 'strength' in strategy:
            # Long-term initiative
            strategy_items.append(html.Div([
                html.H6([
                    html.I(className="fas fa-star me-2"),
                    strategy['strength']
                ], className="fw-bold text-success"),
                html.P(strategy['initiative'], className="mb-2"),                html.Div([
                    dbc.Badge(f"{strategy['count']} customers value this", color="light", className="me-2"),
                    html.Small([
                        html.Strong("Timeline: "),
                        strategy['timeline']
                    ], className="text-muted")
                ], className="d-flex justify-content-between align-items-center"),
                html.Small(strategy['scalability'], className="text-success")
            ], className="mb-3"))
    
    return dbc.Card([
        dbc.CardHeader([
            html.H4(title, className="mb-0"),
            html.Small(timeline, className="text-muted")
        ]),
        dbc.CardBody(strategy_items)
    ], color=color, outline=True, className="h-100")

def create_monitoring_framework_section(monitoring_systems):
    """Create monitoring framework section"""
    return html.Div([
        html.H2([
            html.I(className="fas fa-chart-line me-2"),
            "Monitoring & Evaluation Framework"
        ], className="section-title"),
        
        html.Div([
            dbc.Card([
                dbc.CardBody([
                    html.H5([
                        html.I(className="fas fa-desktop me-2"),
                        system['system']
                    ], className="card-title"),
                    html.P(system['description'], className="card-text"),
                    html.Div([
                        html.Strong("Frequency: "),
                        html.Span(system['frequency'], className="badge bg-info me-3"),
                        html.Strong("Key Metrics: "),
                        html.Span(", ".join(system['kpis']), className="text-muted")
                    ])
                ])
            ], className="mb-3") for system in monitoring_systems
        ])
    ], className="monitoring-framework-section mb-4")

def create_implementation_roadmap_section(recommendations):
    """Create implementation roadmap section"""
    return html.Div([
        html.H2([
            html.I(className="fas fa-road me-2"),
            "Implementation Roadmap"
        ], className="section-title"),
        
        dbc.Card([
            dbc.CardBody([
                html.H5("📅 Recommended Implementation Timeline", className="card-title"),
                
                dbc.Row([
                    dbc.Col([
                        html.Div([
                            html.H6("Phase 1: Immediate (0-30 days)", className="text-danger fw-bold"),
                            html.Ul([
                                html.Li("Address critical safety and cleanliness issues"),
                                html.Li("Implement emergency response protocols"),
                                html.Li("Launch staff training for priority areas"),
                                html.Li("Establish monitoring systems")
                            ])
                        ])
                    ], md=6),
                    
                    dbc.Col([
                        html.Div([
                            html.H6("Phase 2: Short-term (1-6 months)", className="text-warning fw-bold"),
                            html.Ul([
                                html.Li("Roll out comprehensive training programs"),
                                html.Li("Upgrade facilities and infrastructure"),
                                html.Li("Implement quality assurance systems"),
                                html.Li("Launch customer feedback mechanisms")
                            ])
                        ])
                    ], md=6)
                ], className="mb-3"),
                
                dbc.Row([
                    dbc.Col([
                        html.Div([
                            html.H6("Phase 3: Medium-term (6-12 months)", className="text-info fw-bold"),
                            html.Ul([
                                html.Li("Expand successful practices across regions"),
                                html.Li("Develop tourism promotion campaigns"),
                                html.Li("Establish partnership programs"),
                                html.Li("Create certification standards")
                            ])
                        ])
                    ], md=6),
                    
                    dbc.Col([
                        html.Div([
                            html.H6("Phase 4: Long-term (1-2 years)", className="text-success fw-bold"),
                            html.Ul([
                                html.Li("Build sustainable tourism infrastructure"),
                                html.Li("Develop specialized tourism products"),
                                html.Li("Create international marketing strategies"),
                                html.Li("Establish tourism research centers")
                            ])
                        ])
                    ], md=6)
                ])
            ])
        ])
    ], className="implementation-roadmap-section mb-4")

# Callbacks for dynamic chart content
@callback(
    Output('top-complaints-chart', 'children'),
    Input('page-content', 'children'),
    prevent_initial_call=True
)
def update_complaints_chart(_):
    """Update top complaints chart"""
    try:
        df = load_recommendations_data()
        if df.empty:
            return html.P("No data available", className="text-muted")
        
        insights = analyze_key_insights(df)
        if not insights['top_complaints']:
            return html.P("No complaint data available", className="text-muted")
        
        # Create horizontal bar chart for top complaints
        complaints_data = insights['top_complaints'][:8]  # Top 8 complaints
        
        fig = go.Figure(data=[
            go.Bar(
                y=[item[0].title() for item in complaints_data],
                x=[item[1] for item in complaints_data],
                orientation='h',
                marker=dict(
                    color='rgba(239, 68, 68, 0.8)',
                    line=dict(color='rgba(239, 68, 68, 1.0)', width=1)
                ),
                hovertemplate='<b>%{y}</b><br>Mentions: %{x}<extra></extra>'
            )
        ])
        
        fig.update_layout(
            title="Most Frequent Complaints",
            xaxis_title="Number of Mentions",
            yaxis_title="",
            height=400,
            margin=dict(l=20, r=20, t=40, b=20),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        
        return dcc.Graph(figure=fig, config={'displayModeBar': False})
        
    except Exception as e:
        print(f"Error creating complaints chart: {e}")
        return html.P("Error loading complaints data", className="text-danger")

@callback(
    Output('top-strengths-chart', 'children'),
    Input('page-content', 'children'),
    prevent_initial_call=True
)
def update_strengths_chart(_):
    """Update top strengths chart"""
    try:
        df = load_recommendations_data()
        if df.empty:
            return html.P("No data available", className="text-muted")
        
        insights = analyze_key_insights(df)
        if not insights['top_strengths']:
            return html.P("No strengths data available", className="text-muted")
        
        # Create horizontal bar chart for top strengths
        strengths_data = insights['top_strengths'][:8]  # Top 8 strengths
        
        fig = go.Figure(data=[
            go.Bar(
                y=[item[0].title() for item in strengths_data],
                x=[item[1] for item in strengths_data],
                orientation='h',
                marker=dict(
                    color='rgba(16, 185, 129, 0.8)',
                    line=dict(color='rgba(16, 185, 129, 1.0)', width=1)
                ),
                hovertemplate='<b>%{y}</b><br>Mentions: %{x}<extra></extra>'
            )
        ])
        
        fig.update_layout(
            title="Top Tourism Strengths",
            xaxis_title="Number of Mentions",
            yaxis_title="",
            height=400,
            margin=dict(l=20, r=20, t=40, b=20),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        
        return dcc.Graph(figure=fig, config={'displayModeBar': False})
        
    except Exception as e:
        print(f"Error creating strengths chart: {e}")
        return html.P("Error loading strengths data", className="text-danger")
