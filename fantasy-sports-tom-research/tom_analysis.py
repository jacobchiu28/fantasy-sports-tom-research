import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

# Set style for better plots
plt.style.use('default')
sns.set_palette("husl")

def load_results():
    """Load the results CSV"""
    return pd.read_csv('runs/analysis/results.csv')

def create_tom_comparison_plots(df):
    """Create comprehensive ToM analysis plots"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Theory of Mind (ToM) vs Baseline Analysis', fontsize=16, fontweight='bold')
    
    # 1. Total ToM Score Distribution
    ax1 = axes[0, 0]
    tom_data = df[df['variant'] == 'tom']['total_tom_score']
    baseline_data = df[df['variant'] == 'baseline']['total_tom_score']
    
    ax1.hist([baseline_data, tom_data], bins=15, alpha=0.7, 
             label=['Baseline', 'ToM'], color=['orange', 'blue'])
    ax1.set_xlabel('Total ToM Score')
    ax1.set_ylabel('Frequency')
    ax1.set_title('ToM Score Distribution')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. ToM Components Breakdown
    ax2 = axes[0, 1]
    tom_means = df[df['variant'] == 'tom'][['tom_keywords', 'perspective_taking', 
                                           'need_analysis', 'mutual_consideration', 
                                           'reasoning_depth']].mean()
    baseline_means = df[df['variant'] == 'baseline'][['tom_keywords', 'perspective_taking', 
                                                     'need_analysis', 'mutual_consideration', 
                                                     'reasoning_depth']].mean()
    
    x = np.arange(len(tom_means))
    width = 0.35
    ax2.bar(x - width/2, baseline_means, width, label='Baseline', color='orange', alpha=0.7)
    ax2.bar(x + width/2, tom_means, width, label='ToM', color='blue', alpha=0.7)
    ax2.set_xlabel('ToM Components')
    ax2.set_ylabel('Average Score')
    ax2.set_title('ToM Components Breakdown')
    ax2.set_xticks(x)
    ax2.set_xticklabels(['Keywords', 'Perspective', 'Need Analysis', 'Mutual', 'Depth'], rotation=45)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Reasoning Length vs Quality
    ax3 = axes[0, 2]
    scatter_data = df.groupby(['variant', 'reasoning_length'])['total_tom_score'].mean().reset_index()
    for variant in ['tom', 'baseline']:
        subset = scatter_data[scatter_data['variant'] == variant]
        ax3.scatter(subset['reasoning_length'], subset['total_tom_score'], 
                   label=variant, alpha=0.6, s=50)
    ax3.set_xlabel('Reasoning Length (words)')
    ax3.set_ylabel('Total ToM Score')
    ax3.set_title('Reasoning Length vs ToM Quality')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Trade Performance by ToM Score
    ax4 = axes[1, 0]
    df['tom_score_bin'] = pd.cut(df['total_tom_score'], bins=5, labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])
    trade_perf = df.groupby(['tom_score_bin', 'variant'])['mutual_benefit'].mean().unstack()
    trade_perf.plot(kind='bar', ax=ax4, color=['orange', 'blue'], alpha=0.7)
    ax4.set_xlabel('ToM Score Bins')
    ax4.set_ylabel('Mutual Benefit Rate')
    ax4.set_title('Trade Success by ToM Score')
    ax4.legend(['Baseline', 'ToM'])
    ax4.tick_params(axis='x', rotation=45)
    ax4.grid(True, alpha=0.3)
    
    # 5. Model Comparison for ToM
    ax5 = axes[1, 1]
    model_tom = df[df['variant'] == 'tom'].groupby('model')['total_tom_score'].mean()
    model_tom.plot(kind='bar', ax=ax5, color='blue', alpha=0.7)
    ax5.set_xlabel('Model')
    ax5.set_ylabel('Average ToM Score')
    ax5.set_title('ToM Performance by Model')
    ax5.tick_params(axis='x', rotation=45)
    ax5.grid(True, alpha=0.3)
    
    # 6. Trade Balance vs ToM Score
    ax6 = axes[1, 2]
    tom_subset = df[df['variant'] == 'tom']
    baseline_subset = df[df['variant'] == 'baseline']
    ax6.scatter(tom_subset['total_tom_score'], tom_subset['trade_balance'], 
               alpha=0.6, label='ToM', color='blue')
    ax6.scatter(baseline_subset['total_tom_score'], baseline_subset['trade_balance'], 
               alpha=0.6, label='Baseline', color='orange')
    ax6.set_xlabel('Total ToM Score')
    ax6.set_ylabel('Trade Balance')
    ax6.set_title('Trade Balance vs ToM Score')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('runs/analysis/tom_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_model_comparison_plots(df):
    """Create inter-model and intra-model comparison plots"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Model Performance Analysis', fontsize=16, fontweight='bold')
    
    # 1. Inter-model ToM comparison
    ax1 = axes[0, 0]
    model_tom_data = []
    models = df['model'].unique()
    for model in models:
        tom_scores = df[(df['model'] == model) & (df['variant'] == 'tom')]['total_tom_score']
        model_tom_data.append(tom_scores)
    
    ax1.boxplot(model_tom_data, labels=models)
    ax1.set_xlabel('Model')
    ax1.set_ylabel('ToM Score')
    ax1.set_title('ToM Performance Across Models')
    ax1.grid(True, alpha=0.3)
    
    # 2. Intra-model ToM vs Baseline
    ax2 = axes[0, 1]
    for i, model in enumerate(models):
        model_data = df[df['model'] == model]
        tom_mean = model_data[model_data['variant'] == 'tom']['total_tom_score'].mean()
        baseline_mean = model_data[model_data['variant'] == 'baseline']['total_tom_score'].mean()
        
        ax2.plot([i-0.1, i+0.1], [baseline_mean, tom_mean], 'o-', 
                label=model if i == 0 else "", linewidth=2, markersize=8)
        ax2.text(i, max(baseline_mean, tom_mean) + 0.2, model, ha='center', fontsize=10)
    
    ax2.set_xlabel('Model')
    ax2.set_ylabel('Average ToM Score')
    ax2.set_title('ToM vs Baseline by Model')
    ax2.set_xticks(range(len(models)))
    ax2.set_xticklabels(['Baseline→ToM'] * len(models))
    ax2.grid(True, alpha=0.3)
    
    # 3. Trade Success by Model and Variant
    ax3 = axes[1, 0]
    success_by_model = df.groupby(['model', 'variant'])['mutual_benefit'].mean().unstack()
    success_by_model.plot(kind='bar', ax=ax3, color=['orange', 'blue'], alpha=0.7)
    ax3.set_xlabel('Model')
    ax3.set_ylabel('Mutual Benefit Rate')
    ax3.set_title('Trade Success Rate by Model')
    ax3.legend(['Baseline', 'ToM'])
    ax3.tick_params(axis='x', rotation=45)
    ax3.grid(True, alpha=0.3)
    
    # 4. Trade Balance Distribution by Model
    ax4 = axes[1, 1]
    for model in models:
        model_balance = df[df['model'] == model]['trade_balance']
        ax4.hist(model_balance, alpha=0.5, bins=15, label=model, density=True)
    ax4.set_xlabel('Trade Balance')
    ax4.set_ylabel('Density')
    ax4.set_title('Trade Balance Distribution by Model')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('runs/analysis/model_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def generate_statistical_summary(df):
    """Generate comprehensive statistical summary"""
    print("="*80)
    print("COMPREHENSIVE FANTASY FOOTBALL TRADE ANALYSIS")
    print("="*80)
    
    # Overall statistics
    print(f"\n📊 DATASET OVERVIEW")
    print(f"Total trades analyzed: {len(df)}")
    print(f"Models: {', '.join(df['model'].unique())}")
    print(f"Variants: {', '.join(df['variant'].unique())}")
    print(f"Scenarios: {df['scenario_id'].nunique()}")
    print(f"Total metrics captured: {len(df.columns)} columns")
    
    # ToM Analysis
    print(f"\n🧠 THEORY OF MIND ANALYSIS")
    tom_data = df[df['variant'] == 'tom']
    baseline_data = df[df['variant'] == 'baseline']
    
    print(f"ToM vs Baseline Average Scores:")
    print(f"  Total ToM Score: {tom_data['total_tom_score'].mean():.2f} vs {baseline_data['total_tom_score'].mean():.2f}")
    print(f"  Perspective Taking: {tom_data['perspective_taking'].mean():.2f} vs {baseline_data['perspective_taking'].mean():.2f}")
    print(f"  Need Analysis: {tom_data['need_analysis'].mean():.2f} vs {baseline_data['need_analysis'].mean():.2f}")
    print(f"  Mutual Consideration: {tom_data['mutual_consideration'].mean():.2f} vs {baseline_data['mutual_consideration'].mean():.2f}")
    print(f"  Reasoning Depth: {tom_data['reasoning_depth'].mean():.2f} vs {baseline_data['reasoning_depth'].mean():.2f}")
    print(f"  Reasoning Length: {tom_data['reasoning_length'].mean():.1f} vs {baseline_data['reasoning_length'].mean():.1f} words")
    
    # Additional integrated metrics
    print(f"\n🔍 ADVANCED TOM METRICS")
    print(f"  Team Balance: {tom_data['team_balance'].mean():.3f} vs {baseline_data['team_balance'].mean():.3f}")
    print(f"  Justification Score: {tom_data['justification_score'].mean():.1f} vs {baseline_data['justification_score'].mean():.1f}")
    print(f"  Specific Benefits: {tom_data['specific_benefits'].mean():.1f} vs {baseline_data['specific_benefits'].mean():.1f}")
    print(f"  Team A Mentions: {tom_data['team_a_mentions'].mean():.1f} vs {baseline_data['team_a_mentions'].mean():.1f}")
    print(f"  Team B Mentions: {tom_data['team_b_mentions'].mean():.1f} vs {baseline_data['team_b_mentions'].mean():.1f}")
    
    # Trade Performance
    print(f"\n💼 TRADE PERFORMANCE")
    print(f"Mutual Benefit Rate:")
    print(f"  ToM: {tom_data['mutual_benefit'].mean():.1%}")
    print(f"  Baseline: {baseline_data['mutual_benefit'].mean():.1%}")
    print(f"Average Trade Balance:")
    print(f"  ToM: {tom_data['trade_balance'].mean():.3f}")
    print(f"  Baseline: {baseline_data['trade_balance'].mean():.3f}")
    
    # Model-specific analysis
    print(f"\n🤖 MODEL-SPECIFIC ANALYSIS")
    for model in df['model'].unique():
        model_data = df[df['model'] == model]
        model_tom = model_data[model_data['variant'] == 'tom']
        model_baseline = model_data[model_data['variant'] == 'baseline']
        
        print(f"\n{model.upper()}:")
        print(f"  ToM Score: {model_tom['total_tom_score'].mean():.2f} vs {model_baseline['total_tom_score'].mean():.2f}")
        print(f"  Mutual Benefit: {model_tom['mutual_benefit'].mean():.1%} vs {model_baseline['mutual_benefit'].mean():.1%}")
        print(f"  Trade Balance: {model_tom['trade_balance'].mean():.3f} vs {model_baseline['trade_balance'].mean():.3f}")
    
    # Correlations
    print(f"\n📈 KEY CORRELATIONS")
    tom_correlations = tom_data[['total_tom_score', 'mutual_benefit', 'trade_balance', 'reasoning_length']].corr()
    print(f"ToM Score vs Mutual Benefit: {tom_correlations.loc['total_tom_score', 'mutual_benefit']:.3f}")
    print(f"ToM Score vs Trade Balance: {tom_correlations.loc['total_tom_score', 'trade_balance']:.3f}")
    print(f"Reasoning Length vs ToM Score: {tom_correlations.loc['reasoning_length', 'total_tom_score']:.3f}")

def main():
    """Run complete analysis"""
    # Ensure output directory exists
    Path('runs/analysis').mkdir(parents=True, exist_ok=True)
    
    # Load data
    df = load_results()
    
    # Generate analysis
    print("Generating ToM comparison plots...")
    create_tom_comparison_plots(df)
    
    print("Generating model comparison plots...")
    create_model_comparison_plots(df)
    
    print("Generating statistical summary...")
    generate_statistical_summary(df)
    
    print(f"\n✅ Analysis complete! Check runs/analysis/ for:")
    print(f"  - tom_analysis.png: ToM reasoning analysis")
    print(f"  - model_comparison.png: Inter/intra-model comparisons")
    print(f"  - results.csv: Raw data with enhanced metrics")

if __name__ == '__main__':
    main()
