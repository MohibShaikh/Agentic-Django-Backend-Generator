"""
Priority 2: User Experience & Quality Enhancements
=================================================

Building on the successful Priority 1 optimizations (async, caching, validation),
these enhancements focus on improving user experience and code quality.

Key Features:
- Visual diff interface for HITL reviews  
- Batch approve/reject operations
- Intelligent error recovery and self-healing
- Quality metrics and performance monitoring
- Context-aware retry logic with different models
"""

import asyncio
import json
import os
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import difflib
from collections import defaultdict
import statistics

@dataclass
class ErrorPattern:
    """Track and categorize error patterns for intelligent recovery."""
    error_type: str
    error_message: str
    frequency: int
    last_seen: datetime
    successful_recovery_strategy: Optional[str] = None
    alternative_models: List[str] = None

@dataclass
class QualityMetrics:
    """Comprehensive quality metrics for generated code."""
    syntax_score: float
    semantic_score: float
    best_practices_score: float
    security_score: float
    overall_score: float
    lines_of_code: int
    complexity_score: float
    readability_score: float
    timestamp: datetime

@dataclass
class PerformanceMetrics:
    """Track performance across the entire pipeline."""
    total_pipeline_time: float
    cache_hit_rate: float
    auto_approval_rate: float
    error_rate: float
    average_quality_score: float
    agent_performance: Dict[str, Dict[str, float]]
    timestamp: datetime

class VisualDiffInterface:
    """Enhanced visual diff interface for HITL reviews."""
    
    def __init__(self):
        self.review_history = []
    
    def generate_diff_view(self, original: str, modified: str, filename: str) -> str:
        """Generate a visual diff between original and modified content."""
        diff = list(difflib.unified_diff(
            original.splitlines(keepends=True),
            modified.splitlines(keepends=True),
            fromfile=f"original/{filename}",
            tofile=f"modified/{filename}",
            lineterm=""
        ))
        
        if not diff:
            return "✅ No changes detected"
        
        # Format diff with colors and better readability
        formatted_diff = []
        formatted_diff.append("📄 " + "="*60)
        formatted_diff.append(f"🔍 DIFF VIEW: {filename}")
        formatted_diff.append("="*60)
        
        for line in diff:
            if line.startswith('+++') or line.startswith('---'):
                formatted_diff.append(f"📂 {line.strip()}")
            elif line.startswith('@@'):
                formatted_diff.append(f"📍 {line.strip()}")
            elif line.startswith('+'):
                formatted_diff.append(f"✅ {line}")
            elif line.startswith('-'):
                formatted_diff.append(f"❌ {line}")
            else:
                formatted_diff.append(f"   {line}")
        
        return "\n".join(formatted_diff)
    
    def create_review_summary(self, files_data: Dict[str, Any]) -> str:
        """Create a comprehensive review summary for batch operations."""
        summary = []
        summary.append("🎯 " + "="*60)
        summary.append("📋 BATCH REVIEW SUMMARY")
        summary.append("="*60)
        
        auto_approved = []
        needs_review = []
        errors = []
        
        for filename, data in files_data.items():
            validation = data.get('validation')
            if validation.auto_approve:
                auto_approved.append(filename)
            elif validation.errors:
                errors.append((filename, len(validation.errors)))
            else:
                needs_review.append(filename)
        
        summary.append(f"✅ Auto-approved: {len(auto_approved)} files")
        if auto_approved:
            for file in auto_approved:
                summary.append(f"   • {file}")
        
        summary.append(f"❌ Errors: {len(errors)} files")
        if errors:
            for file, error_count in errors:
                summary.append(f"   • {file} ({error_count} errors)")
        
        summary.append(f"👁️  Manual review needed: {len(needs_review)} files")
        if needs_review:
            for file in needs_review:
                summary.append(f"   • {file}")
        
        return "\n".join(summary)

class BatchOperationsManager:
    """Handle batch approve/reject operations for improved workflow."""
    
    def __init__(self):
        self.batch_decisions = {}
        self.review_patterns = defaultdict(int)
    
    async def batch_review_interface(self, files_data: Dict[str, Any]) -> Dict[str, str]:
        """Present batch review interface with smart defaults."""
        print("\n" + "🚀 " + "="*60)
        print("⚡ BATCH REVIEW INTERFACE")
        print("="*60)
        
        # Categorize files
        auto_approved = []
        high_quality = []
        medium_quality = []
        low_quality = []
        errors = []
        
        for filename, data in files_data.items():
            validation = data['validation']
            quality = validation.quality_score
            
            if validation.auto_approve:
                auto_approved.append(filename)
            elif validation.errors:
                errors.append(filename)
            elif quality >= 0.7:
                high_quality.append(filename)
            elif quality >= 0.5:
                medium_quality.append(filename)
            else:
                low_quality.append(filename)
        
        # Display categorized summary
        print(f"✅ Auto-approved ({len(auto_approved)}): {', '.join(auto_approved)}")
        print(f"🟢 High quality ({len(high_quality)}): {', '.join(high_quality)}")
        print(f"🟡 Medium quality ({len(medium_quality)}): {', '.join(medium_quality)}")
        print(f"🔴 Low quality ({len(low_quality)}): {', '.join(low_quality)}")
        print(f"❌ Errors ({len(errors)}): {', '.join(errors)}")
        
        print("\n📋 Batch Operations:")
        print("  [1] Approve all high quality")
        print("  [2] Approve high + medium quality")
        print("  [3] Review each file individually")
        print("  [4] Skip all problematic files")
        print("  [5] Custom selection")
        
        try:
            choice = input("\nSelect batch operation (1-5): ").strip()
        except EOFError:
            choice = "1"  # Default to approve high quality
        
        decisions = {}
        
        # Auto-approved files
        for filename in auto_approved:
            decisions[filename] = "approved"
        
        if choice == "1":
            # Approve high quality only
            for filename in high_quality:
                decisions[filename] = "approved"
            for filename in medium_quality + low_quality + errors:
                decisions[filename] = "skipped"
                
        elif choice == "2":
            # Approve high + medium quality
            for filename in high_quality + medium_quality:
                decisions[filename] = "approved"
            for filename in low_quality + errors:
                decisions[filename] = "skipped"
                
        elif choice == "3":
            # Individual review
            return await self._individual_review(files_data)
            
        elif choice == "4":
            # Skip all problematic
            for filename in high_quality + medium_quality:
                decisions[filename] = "approved"
            for filename in low_quality + errors:
                decisions[filename] = "skipped"
                
        elif choice == "5":
            # Custom selection
            return await self._custom_selection(files_data)
        
        print(f"\n✅ Batch operation complete: {len([d for d in decisions.values() if d == 'approved'])} approved, {len([d for d in decisions.values() if d == 'skipped'])} skipped")
        
        return decisions
    
    async def _individual_review(self, files_data: Dict[str, Any]) -> Dict[str, str]:
        """Handle individual file review with enhanced interface."""
        decisions = {}
        diff_interface = VisualDiffInterface()
        
        for filename, data in files_data.items():
            validation = data['validation']
            content = data['content']
            
            if validation.auto_approve:
                decisions[filename] = "approved"
                continue
            
            print(f"\n" + "🔍 " + "="*60)
            print(f"📄 REVIEWING: {filename}")
            print(f"🎯 Quality Score: {validation.quality_score:.2f}")
            print("="*60)
            
            # Show validation details
            if validation.errors:
                print(f"❌ ERRORS ({len(validation.errors)}):")
                for i, error in enumerate(validation.errors, 1):
                    print(f"   {i}. {error}")
            
            if validation.warnings:
                print(f"⚠️  WARNINGS ({len(validation.warnings)}):")
                for i, warning in enumerate(validation.warnings, 1):
                    print(f"   {i}. {warning}")
            
            if validation.suggestions:
                print(f"💡 SUGGESTIONS ({len(validation.suggestions)}):")
                for i, suggestion in enumerate(validation.suggestions, 1):
                    print(f"   {i}. {suggestion}")
            
            # Show content preview
            print(f"\n📖 Content preview (first 300 chars):")
            print("-" * 40)
            print(content[:300] + ("..." if len(content) > 300 else ""))
            print("-" * 40)
            
            # Enhanced options
            print(f"\n⚡ Options for {filename}:")
            print("  [a] Approve")
            print("  [r] Reject") 
            print("  [s] Skip")
            print("  [v] View full content")
            print("  [q] Quick approve remaining high-quality files")
            
            try:
                decision = input(f"Decision: ").lower().strip()
            except EOFError:
                decision = "a"
            
            if decision == 'q':
                # Quick approve remaining high-quality files
                decisions[filename] = "approved"
                for remaining_file, remaining_data in files_data.items():
                    if remaining_file not in decisions:
                        if remaining_data['validation'].quality_score >= 0.7:
                            decisions[remaining_file] = "approved"
                        else:
                            decisions[remaining_file] = "skipped"
                break
            elif decision == 'v':
                print(f"\n📄 Full content of {filename}:")
                print("=" * 60)
                print(content)
                print("=" * 60)
                # Ask again after showing content
                decision = input(f"Decision for {filename}: ").lower().strip()
            
            if decision == 'r':
                decisions[filename] = "rejected"
            elif decision == 's':
                decisions[filename] = "skipped"
            else:
                decisions[filename] = "approved"
        
        return decisions
    
    async def _custom_selection(self, files_data: Dict[str, Any]) -> Dict[str, str]:
        """Handle custom file selection for batch operations."""
        files = list(files_data.keys())
        print(f"\n📋 Custom Selection - Available files:")
        for i, filename in enumerate(files, 1):
            quality = files_data[filename]['validation'].quality_score
            status = "✅" if files_data[filename]['validation'].auto_approve else "🔍"
            print(f"  [{i}] {status} {filename} (Q: {quality:.2f})")
        
        print(f"\n💡 Enter file numbers to approve (e.g., '1,3,5' or '1-4'):")
        print(f"   Or 'all' to approve all, 'none' to skip all")
        
        try:
            selection = input("Selection: ").strip()
        except EOFError:
            selection = "all"
        
        decisions = {}
        
        if selection == "all":
            for filename in files:
                decisions[filename] = "approved"
        elif selection == "none":
            for filename in files:
                decisions[filename] = "skipped"
        else:
            # Parse selection
            selected_indices = set()
            for part in selection.split(','):
                if '-' in part:
                    start, end = map(int, part.split('-'))
                    selected_indices.update(range(start, end + 1))
                else:
                    selected_indices.add(int(part))
            
            for i, filename in enumerate(files, 1):
                if i in selected_indices:
                    decisions[filename] = "approved"
                else:
                    decisions[filename] = "skipped"
        
        return decisions

class IntelligentErrorRecovery:
    """Self-healing system with intelligent error recovery."""
    
    def __init__(self):
        self.error_patterns = {}
        self.recovery_strategies = {
            "rate_limit": self._handle_rate_limit,
            "syntax_error": self._handle_syntax_error,
            "api_error": self._handle_api_error,
            "validation_error": self._handle_validation_error
        }
        self.alternative_models = [
            "qwen/qwen3-coder:free",
            "deepseek/deepseek-r1-0528:free", 
            "meta-llama/llama-3.2-3b-instruct:free",
            "microsoft/phi-3-mini-128k-instruct:free"
        ]
    
    async def recover_from_error(self, error: Exception, context: Dict[str, Any]) -> Optional[str]:
        """Attempt intelligent error recovery based on error patterns."""
        error_type = self._classify_error(error)
        error_key = f"{error_type}:{str(error)[:100]}"
        
        # Track error pattern
        if error_key in self.error_patterns:
            self.error_patterns[error_key].frequency += 1
            self.error_patterns[error_key].last_seen = datetime.now()
        else:
            self.error_patterns[error_key] = ErrorPattern(
                error_type=error_type,
                error_message=str(error)[:200],
                frequency=1,
                last_seen=datetime.now()
            )
        
        # Attempt recovery based on error type
        if error_type in self.recovery_strategies:
            recovery_func = self.recovery_strategies[error_type]
            return await recovery_func(error, context)
        
        return None
    
    def _classify_error(self, error: Exception) -> str:
        """Classify error type for appropriate recovery strategy."""
        error_str = str(error).lower()
        
        if "rate limit" in error_str or "429" in error_str:
            return "rate_limit"
        elif "syntax" in error_str or "compilation" in error_str:
            return "syntax_error" 
        elif "api" in error_str or "connection" in error_str:
            return "api_error"
        elif "validation" in error_str:
            return "validation_error"
        else:
            return "unknown"
    
    async def _handle_rate_limit(self, error: Exception, context: Dict[str, Any]) -> Optional[str]:
        """Handle rate limit errors with exponential backoff and model switching."""
        print(f"🔄 Rate limit detected. Attempting recovery...")
        
        # Try alternative model
        current_model = context.get('model', 'qwen/qwen3-coder:free')
        alternative_models = [m for m in self.alternative_models if m != current_model]
        
        if alternative_models:
            new_model = alternative_models[0]
            print(f"🔄 Switching from {current_model} to {new_model}")
            context['model'] = new_model
            
            # Wait before retry
            await asyncio.sleep(2)
            return f"Switched to alternative model: {new_model}"
        
        # If no alternatives, suggest waiting
        print(f"⏰ No alternative models available. Consider waiting or adding credits.")
        return None
    
    async def _handle_syntax_error(self, error: Exception, context: Dict[str, Any]) -> Optional[str]:
        """Handle syntax errors with simplified prompts."""
        print(f"🔧 Syntax error detected. Attempting simpler generation...")
        
        # Simplify the prompt
        original_prompt = context.get('prompt', '')
        simplified_prompt = self._simplify_prompt(original_prompt)
        context['prompt'] = simplified_prompt
        
        return "Simplified prompt for better syntax"
    
    async def _handle_api_error(self, error: Exception, context: Dict[str, Any]) -> Optional[str]:
        """Handle API errors with connection retry."""
        print(f"🌐 API error detected. Retrying with exponential backoff...")
        
        retry_count = context.get('retry_count', 0)
        if retry_count < 3:
            wait_time = 2 ** retry_count
            await asyncio.sleep(wait_time)
            context['retry_count'] = retry_count + 1
            return f"Retried after {wait_time}s (attempt {retry_count + 1})"
        
        return None
    
    async def _handle_validation_error(self, error: Exception, context: Dict[str, Any]) -> Optional[str]:
        """Handle validation errors with prompt enhancement."""
        print(f"✅ Validation error detected. Enhancing prompt with specific constraints...")
        
        # Add validation-specific instructions
        original_prompt = context.get('prompt', '')
        enhanced_prompt = self._enhance_prompt_for_validation(original_prompt)
        context['prompt'] = enhanced_prompt
        
        return "Enhanced prompt with validation constraints"
    
    def _simplify_prompt(self, prompt: str) -> str:
        """Simplify prompt to reduce syntax errors."""
        simplified = prompt.replace("complex", "simple")
        simplified = simplified.replace("advanced", "basic")
        simplified += "\n\nIMPORTANT: Generate only simple, valid Python code without complex features."
        return simplified
    
    def _enhance_prompt_for_validation(self, prompt: str) -> str:
        """Enhance prompt with validation-specific constraints."""
        enhanced = prompt + "\n\nVALIDATION REQUIREMENTS:\n"
        enhanced += "- Ensure all imports are included\n"
        enhanced += "- Use proper Django/DRF patterns\n" 
        enhanced += "- Include all necessary classes and methods\n"
        enhanced += "- Follow PEP8 style guidelines\n"
        return enhanced

class PerformanceDashboard:
    """Real-time performance monitoring and analytics dashboard."""
    
    def __init__(self):
        self.metrics_history = []
        self.quality_trends = []
        self.performance_baselines = {
            'pipeline_time': 30.0,  # seconds
            'cache_hit_rate': 0.3,  # 30%
            'auto_approval_rate': 0.8,  # 80%
            'error_rate': 0.1,  # 10%
            'quality_score': 0.8  # 80%
        }
    
    def record_metrics(self, metrics: PerformanceMetrics) -> None:
        """Record performance metrics for analysis."""
        self.metrics_history.append(metrics)
        
        # Keep only last 100 runs for analysis
        if len(self.metrics_history) > 100:
            self.metrics_history = self.metrics_history[-100:]
    
    def generate_performance_report(self) -> str:
        """Generate comprehensive performance report."""
        if not self.metrics_history:
            return "📊 No performance data available yet."
        
        latest = self.metrics_history[-1]
        
        report = []
        report.append("📊 " + "="*60)
        report.append("🚀 PERFORMANCE DASHBOARD")
        report.append("="*60)
        
        # Current Performance
        report.append("📈 Current Performance:")
        report.append(f"   ⏱️  Pipeline Time: {latest.total_pipeline_time:.2f}s")
        report.append(f"   💾 Cache Hit Rate: {latest.cache_hit_rate:.1%}")
        report.append(f"   ✅ Auto-approval Rate: {latest.auto_approval_rate:.1%}")
        report.append(f"   ❌ Error Rate: {latest.error_rate:.1%}")
        report.append(f"   🎯 Avg Quality Score: {latest.average_quality_score:.2f}")
        
        # Performance vs Baselines
        report.append("\n🎯 Performance vs Baselines:")
        report.append(self._compare_to_baseline("Pipeline Time", latest.total_pipeline_time, self.performance_baselines['pipeline_time'], lower_is_better=True))
        report.append(self._compare_to_baseline("Cache Hit Rate", latest.cache_hit_rate, self.performance_baselines['cache_hit_rate']))
        report.append(self._compare_to_baseline("Auto-approval Rate", latest.auto_approval_rate, self.performance_baselines['auto_approval_rate']))
        report.append(self._compare_to_baseline("Error Rate", latest.error_rate, self.performance_baselines['error_rate'], lower_is_better=True))
        report.append(self._compare_to_baseline("Quality Score", latest.average_quality_score, self.performance_baselines['quality_score']))
        
        # Trends (if we have enough data)
        if len(self.metrics_history) >= 5:
            report.append("\n📈 Trends (last 5 runs):")
            recent_metrics = self.metrics_history[-5:]
            
            pipeline_times = [m.total_pipeline_time for m in recent_metrics]
            quality_scores = [m.average_quality_score for m in recent_metrics]
            
            report.append(f"   ⏱️  Pipeline Time: {self._trend_arrow(pipeline_times)} (avg: {statistics.mean(pipeline_times):.1f}s)")
            report.append(f"   🎯 Quality Score: {self._trend_arrow(quality_scores)} (avg: {statistics.mean(quality_scores):.2f})")
        
        # Agent Performance
        if latest.agent_performance:
            report.append("\n🤖 Agent Performance:")
            for agent_name, metrics in latest.agent_performance.items():
                time_taken = metrics.get('time', 0)
                quality = metrics.get('quality', 0)
                cached = "💾" if metrics.get('cached', False) else "🔄"
                report.append(f"   {cached} {agent_name}: {time_taken:.2f}s (Q: {quality:.2f})")
        
        # Recommendations
        report.append("\n💡 Recommendations:")
        recommendations = self._generate_recommendations(latest)
        for rec in recommendations:
            report.append(f"   • {rec}")
        
        report.append("="*60)
        
        return "\n".join(report)
    
    def _compare_to_baseline(self, metric_name: str, current: float, baseline: float, lower_is_better: bool = False) -> str:
        """Compare current metric to baseline with visual indicators."""
        if lower_is_better:
            if current <= baseline * 0.8:
                status = "🟢 Excellent"
            elif current <= baseline:
                status = "🟡 Good"
            else:
                status = "🔴 Needs Improvement"
        else:
            if current >= baseline * 1.2:
                status = "🟢 Excellent"
            elif current >= baseline:
                status = "🟡 Good"
            else:
                status = "🔴 Needs Improvement"
        
        percentage = ((current - baseline) / baseline) * 100
        return f"   {status} {metric_name}: {current:.2f} vs {baseline:.2f} ({percentage:+.1f}%)"
    
    def _trend_arrow(self, values: List[float]) -> str:
        """Generate trend arrow based on recent values."""
        if len(values) < 2:
            return "➡️"
        
        recent_avg = statistics.mean(values[-2:])
        earlier_avg = statistics.mean(values[:-2]) if len(values) > 2 else values[0]
        
        if recent_avg > earlier_avg * 1.05:
            return "📈"
        elif recent_avg < earlier_avg * 0.95:
            return "📉"
        else:
            return "➡️"
    
    def _generate_recommendations(self, metrics: PerformanceMetrics) -> List[str]:
        """Generate performance improvement recommendations."""
        recommendations = []
        
        if metrics.total_pipeline_time > self.performance_baselines['pipeline_time']:
            recommendations.append("Consider optimizing agent prompts or using faster models")
        
        if metrics.cache_hit_rate < self.performance_baselines['cache_hit_rate']:
            recommendations.append("ERD variations detected - consider normalizing input format")
        
        if metrics.auto_approval_rate < self.performance_baselines['auto_approval_rate']:
            recommendations.append("Quality scores low - review and improve generation prompts")
        
        if metrics.error_rate > self.performance_baselines['error_rate']:
            recommendations.append("High error rate - implement additional error recovery strategies")
        
        if metrics.average_quality_score < self.performance_baselines['quality_score']:
            recommendations.append("Consider using higher-quality models for critical files")
        
        if not recommendations:
            recommendations.append("Performance is excellent! Consider documenting current setup as baseline.")
        
        return recommendations

# Integration class for Priority 2 enhancements
class Priority2EnhancedPlannerAgent:
    """
    Enhanced PlannerAgent with Priority 2 optimizations:
    - Visual diff interface
    - Batch operations  
    - Intelligent error recovery
    - Performance monitoring
    """
    
    def __init__(self, base_planner_agent):
        self.base_agent = base_planner_agent
        self.diff_interface = VisualDiffInterface()
        self.batch_manager = BatchOperationsManager()
        self.error_recovery = IntelligentErrorRecovery()
        self.dashboard = PerformanceDashboard()
        
    async def enhanced_hitl_workflow(self, files_data: Dict[str, Any]) -> Dict[str, str]:
        """Enhanced HITL workflow with batch operations and visual diffs."""
        print("\n🚀 " + "="*60)
        print("⚡ ENHANCED HITL WORKFLOW (Priority 2)")
        print("="*60)
        
        # Show performance summary
        performance_report = self.dashboard.generate_performance_report()
        if "No performance data" not in performance_report:
            print(performance_report)
        
        # Use batch operations manager
        decisions = await self.batch_manager.batch_review_interface(files_data)
        
        return decisions
    
    async def enhanced_error_recovery(self, error: Exception, context: Dict[str, Any]) -> Optional[str]:
        """Enhanced error recovery with intelligent strategies."""
        return await self.error_recovery.recover_from_error(error, context)
    
    def record_performance_metrics(self, pipeline_metrics: Dict[str, Any]) -> None:
        """Record performance metrics for dashboard."""
        metrics = PerformanceMetrics(
            total_pipeline_time=pipeline_metrics.get('total_time', 0),
            cache_hit_rate=pipeline_metrics.get('cache_hit_rate', 0),
            auto_approval_rate=pipeline_metrics.get('auto_approval_rate', 0),
            error_rate=pipeline_metrics.get('error_rate', 0),
            average_quality_score=pipeline_metrics.get('average_quality', 0),
            agent_performance=pipeline_metrics.get('agent_performance', {}),
            timestamp=datetime.now()
        )
        
        self.dashboard.record_metrics(metrics)

# Example usage
async def demo_priority2_enhancements():
    """Demonstrate Priority 2 enhancements."""
    print("🚀 Priority 2 Enhancements Demo")
    print("="*50)
    
    # Mock data for demonstration
    mock_files_data = {
        "models.py": {
            "content": "from django.db import models\n\nclass User(models.Model):\n    name = models.CharField(max_length=100)",
            "validation": type('MockValidation', (), {
                "quality_score": 0.85,
                "auto_approve": True,
                "errors": [],
                "warnings": [],
                "suggestions": ["Consider adding __str__ method"]
            })()
        },
        "serializers.py": {
            "content": "from rest_framework import serializers\n\nclass UserSerializer(serializers.ModelSerializer):",
            "validation": type('MockValidation', (), {
                "quality_score": 0.65,
                "auto_approve": False,
                "errors": ["Missing Meta class"],
                "warnings": ["Incomplete serializer"],
                "suggestions": ["Add fields specification"]
            })()
        }
    }
    
    # Demonstrate batch operations
    batch_manager = BatchOperationsManager()
    decisions = await batch_manager.batch_review_interface(mock_files_data)
    print(f"\nBatch decisions: {decisions}")
    
    # Demonstrate performance dashboard
    dashboard = PerformanceDashboard()
    mock_metrics = PerformanceMetrics(
        total_pipeline_time=15.5,
        cache_hit_rate=0.6,
        auto_approval_rate=0.75,
        error_rate=0.05,
        average_quality_score=0.82,
        agent_performance={
            "ModelAgent": {"time": 3.2, "quality": 0.85, "cached": True},
            "SerializerAgent": {"time": 4.1, "quality": 0.65, "cached": False}
        },
        timestamp=datetime.now()
    )
    
    dashboard.record_metrics(mock_metrics)
    report = dashboard.generate_performance_report()
    print(f"\n{report}")

if __name__ == "__main__":
    asyncio.run(demo_priority2_enhancements()) 