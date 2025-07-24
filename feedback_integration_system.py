#!/usr/bin/env python3
"""
Developer Feedback Integration System
====================================

🔧 SMART LEARNING FROM DEVELOPER FEEDBACK

Features:
✅ Inline Comment Parsing: Detects # FIX:, # TODO:, # IMPROVE: comments
✅ Learning from Corrections: Stores feedback in vector DB
✅ Prompt Fine-tuning: Adapts generation based on user feedback
✅ Pattern Recognition: Learns common fix patterns
✅ Context-Aware Suggestions: Provides intelligent recommendations
✅ Continuous Improvement: Gets better with each interaction

Usage:
- Add inline comments like # FIX: make this async
- System learns from rejections and edits
- Prompts automatically improve over time
"""

import asyncio
import json
import os
import re
import time
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Any, Set, Tuple
from dataclasses import dataclass, field
from collections import defaultdict, Counter
import sqlite3
from datetime import datetime, timedelta
import difflib

# Vector storage for embeddings (simplified implementation)
try:
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    VECTOR_AVAILABLE = True
except ImportError:
    VECTOR_AVAILABLE = False
    print("📝 Vector features disabled. Install: pip install numpy scikit-learn")

@dataclass
class FeedbackEntry:
    """Represents a single piece of developer feedback."""
    id: str
    timestamp: datetime
    file_type: str  # models, views, serializers, etc.
    original_code: str
    corrected_code: Optional[str] = None
    inline_comments: List[str] = field(default_factory=list)
    rejection_reason: Optional[str] = None
    erd_context: Optional[Dict] = None
    satisfaction_score: float = 0.0  # 0-1 scale
    tags: Set[str] = field(default_factory=set)
    
    def __post_init__(self):
        if not self.id:
            self.id = hashlib.md5(f"{self.timestamp}{self.file_type}{self.original_code[:100]}".encode()).hexdigest()[:12]

@dataclass  
class ImprovementPattern:
    """Represents a learned improvement pattern."""
    pattern_id: str
    trigger_keywords: List[str]
    improvement_template: str
    confidence_score: float
    usage_count: int = 0
    success_rate: float = 0.0

class InlineCommentParser:
    """Parse and understand inline developer comments."""
    
    COMMENT_PATTERNS = {
        'fix': r'#\s*FIX[:\s]+(.*?)(?:\n|$)',
        'todo': r'#\s*TODO[:\s]+(.*?)(?:\n|$)', 
        'improve': r'#\s*IMPROVE[:\s]+(.*?)(?:\n|$)',
        'bug': r'#\s*BUG[:\s]+(.*?)(?:\n|$)',
        'performance': r'#\s*PERF[:\s]+(.*?)(?:\n|$)',
        'security': r'#\s*SEC[:\s]+(.*?)(?:\n|$)',
        'async': r'#\s*ASYNC[:\s]+(.*?)(?:\n|$)',
        'optimize': r'#\s*OPT[:\s]+(.*?)(?:\n|$)'
    }
    
    def parse_comments(self, code: str) -> Dict[str, List[str]]:
        """Extract all inline comments from code."""
        comments = defaultdict(list)
        
        for comment_type, pattern in self.COMMENT_PATTERNS.items():
            matches = re.findall(pattern, code, re.IGNORECASE | re.MULTILINE)
            for match in matches:
                comments[comment_type].append(match.strip())
        
        return dict(comments)
    
    def extract_improvement_suggestions(self, comments: Dict[str, List[str]]) -> List[str]:
        """Convert comments into actionable improvement suggestions."""
        suggestions = []
        
        for comment_type, comment_list in comments.items():
            for comment in comment_list:
                if comment_type == 'fix':
                    suggestions.append(f"Fix required: {comment}")
                elif comment_type == 'async':
                    suggestions.append(f"Make asynchronous: {comment}")
                elif comment_type == 'performance':
                    suggestions.append(f"Performance optimization: {comment}")
                elif comment_type == 'security':
                    suggestions.append(f"Security improvement: {comment}")
                else:
                    suggestions.append(f"General improvement ({comment_type}): {comment}")
        
        return suggestions

class FeedbackStorage:
    """Store and retrieve developer feedback for learning."""
    
    def __init__(self, db_path: str = ".feedback_db.sqlite"):
        self.db_path = db_path
        self.vectorizer = TfidfVectorizer(max_features=1000) if VECTOR_AVAILABLE else None
        self._init_database()
    
    def _init_database(self):
        """Initialize SQLite database for feedback storage."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS feedback (
                id TEXT PRIMARY KEY,
                timestamp TEXT,
                file_type TEXT,
                original_code TEXT,
                corrected_code TEXT,
                inline_comments TEXT,
                rejection_reason TEXT,
                erd_context TEXT,
                satisfaction_score REAL,
                tags TEXT
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS patterns (
                pattern_id TEXT PRIMARY KEY,
                trigger_keywords TEXT,
                improvement_template TEXT,
                confidence_score REAL,
                usage_count INTEGER,
                success_rate REAL
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def store_feedback(self, feedback: FeedbackEntry):
        """Store feedback entry in database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO feedback 
            (id, timestamp, file_type, original_code, corrected_code, 
             inline_comments, rejection_reason, erd_context, 
             satisfaction_score, tags)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            feedback.id,
            feedback.timestamp.isoformat(),
            feedback.file_type,
            feedback.original_code,
            feedback.corrected_code,
            json.dumps(feedback.inline_comments),
            feedback.rejection_reason,
            json.dumps(feedback.erd_context) if feedback.erd_context else None,
            feedback.satisfaction_score,
            json.dumps(list(feedback.tags))
        ))
        
        conn.commit()
        conn.close()
    
    def get_similar_feedback(self, code: str, file_type: str, limit: int = 5) -> List[FeedbackEntry]:
        """Find similar feedback entries using vector similarity."""
        if not VECTOR_AVAILABLE:
            return self._get_similar_feedback_simple(code, file_type, limit)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM feedback 
            WHERE file_type = ? 
            ORDER BY timestamp DESC 
            LIMIT 50
        ''', (file_type,))
        
        rows = cursor.fetchall()
        conn.close()
        
        if not rows:
            return []
        
        # Use TF-IDF for similarity
        codes = [code] + [row[3] for row in rows]  # original_code is at index 3
        try:
            tfidf_matrix = self.vectorizer.fit_transform(codes)
            similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
            
            similar_indices = similarities.argsort()[-limit:][::-1]
            
            similar_feedback = []
            for idx in similar_indices:
                if similarities[idx] > 0.1:  # Minimum similarity threshold
                    row = rows[idx]
                    feedback = self._row_to_feedback(row)
                    similar_feedback.append(feedback)
            
            return similar_feedback
        except Exception as e:
            print(f"⚠️ Vector similarity failed: {e}")
            return self._get_similar_feedback_simple(code, file_type, limit)
    
    def _get_similar_feedback_simple(self, code: str, file_type: str, limit: int) -> List[FeedbackEntry]:
        """Simple keyword-based similarity matching."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM feedback 
            WHERE file_type = ? 
            ORDER BY timestamp DESC 
            LIMIT ?
        ''', (file_type, limit))
        
        rows = cursor.fetchall()
        conn.close()
        
        return [self._row_to_feedback(row) for row in rows]
    
    def _row_to_feedback(self, row) -> FeedbackEntry:
        """Convert database row to FeedbackEntry."""
        return FeedbackEntry(
            id=row[0],
            timestamp=datetime.fromisoformat(row[1]),
            file_type=row[2],
            original_code=row[3],
            corrected_code=row[4],
            inline_comments=json.loads(row[5]) if row[5] else [],
            rejection_reason=row[6],
            erd_context=json.loads(row[7]) if row[7] else None,
            satisfaction_score=row[8] or 0.0,
            tags=set(json.loads(row[9])) if row[9] else set()
        )

class PatternLearner:
    """Learn improvement patterns from developer feedback."""
    
    def __init__(self, storage: FeedbackStorage):
        self.storage = storage
        self.learned_patterns = {}
    
    def analyze_feedback_patterns(self) -> Dict[str, ImprovementPattern]:
        """Analyze stored feedback to identify improvement patterns."""
        conn = sqlite3.connect(self.storage.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM feedback 
            WHERE corrected_code IS NOT NULL 
            ORDER BY timestamp DESC 
            LIMIT 100
        ''')
        
        rows = cursor.fetchall()
        conn.close()
        
        patterns = {}
        
        for row in rows:
            feedback = self.storage._row_to_feedback(row)
            
            if feedback.corrected_code:
                diff_patterns = self._extract_diff_patterns(
                    feedback.original_code, 
                    feedback.corrected_code
                )
                
                for pattern in diff_patterns:
                    pattern_id = hashlib.md5(pattern['trigger'].encode()).hexdigest()[:8]
                    
                    if pattern_id in patterns:
                        patterns[pattern_id].usage_count += 1
                    else:
                        patterns[pattern_id] = ImprovementPattern(
                            pattern_id=pattern_id,
                            trigger_keywords=pattern['keywords'],
                            improvement_template=pattern['template'],
                            confidence_score=pattern['confidence'],
                            usage_count=1
                        )
        
        self.learned_patterns = patterns
        return patterns
    
    def _extract_diff_patterns(self, original: str, corrected: str) -> List[Dict]:
        """Extract patterns from code differences."""
        patterns = []
        
        # Use difflib to find differences
        diff = list(difflib.unified_diff(
            original.splitlines(),
            corrected.splitlines(),
            lineterm=''
        ))
        
        if len(diff) < 4:  # Skip minimal diffs
            return patterns
        
        # Analyze common patterns
        added_lines = [line[1:] for line in diff if line.startswith('+') and not line.startswith('+++')]
        removed_lines = [line[1:] for line in diff if line.startswith('-') and not line.startswith('---')]
        
        # Pattern: Adding async/await
        if any('async ' in line or 'await ' in line for line in added_lines):
            patterns.append({
                'trigger': 'async_conversion',
                'keywords': ['def ', 'function'],
                'template': 'Convert to async function and add await for database calls',
                'confidence': 0.8
            })
        
        # Pattern: Adding error handling
        if any('try:' in line or 'except' in line for line in added_lines):
            patterns.append({
                'trigger': 'error_handling',
                'keywords': ['def ', 'request', 'response'],
                'template': 'Add proper error handling with try/except blocks',
                'confidence': 0.9
            })
        
        # Pattern: Adding validation
        if any('validate' in line.lower() or 'check' in line.lower() for line in added_lines):
            patterns.append({
                'trigger': 'validation',
                'keywords': ['serializer', 'form', 'model'],
                'template': 'Add input validation and data checks',
                'confidence': 0.85
            })
        
        return patterns

class AdaptivePromptGenerator:
    """Generate improved prompts based on learned feedback patterns."""
    
    def __init__(self, storage: FeedbackStorage, pattern_learner: PatternLearner):
        self.storage = storage
        self.pattern_learner = pattern_learner
        self.base_prompts = self._load_base_prompts()
    
    def _load_base_prompts(self) -> Dict[str, str]:
        """Load base prompts for different file types."""
        return {
            'models': """
Generate Django models with the following requirements:
- Include all necessary imports
- Add proper field types and relationships
- Include __str__ methods and Meta classes
- Follow Django best practices
""",
            'views': """
Generate Django REST Framework views with the following requirements:
- Use ModelViewSet for CRUD operations
- Include proper authentication and permissions
- Add error handling and validation
- Follow DRF best practices
""",
            'serializers': """
Generate Django REST Framework serializers with the following requirements:
- Use ModelSerializer for each model
- Include all necessary fields
- Add custom validation methods
- Handle relationships properly
"""
        }
    
    def generate_adaptive_prompt(self, file_type: str, erd: Dict, context: Dict = None) -> str:
        """Generate an adaptive prompt based on learned patterns."""
        base_prompt = self.base_prompts.get(file_type, "Generate code for the given requirements.")
        
        # Get similar feedback to learn from
        context_code = json.dumps(erd, indent=2)
        similar_feedback = self.storage.get_similar_feedback(context_code, file_type, limit=3)
        
        # Learn from patterns
        patterns = self.pattern_learner.analyze_feedback_patterns()
        
        # Build adaptive improvements
        improvements = []
        
        if similar_feedback:
            common_issues = self._analyze_common_issues(similar_feedback)
            if common_issues:
                improvements.extend([
                    f"⚠️  LEARNED FROM FEEDBACK: {issue}" for issue in common_issues
                ])
        
        if patterns:
            relevant_patterns = self._get_relevant_patterns(patterns, file_type)
            if relevant_patterns:
                improvements.extend([
                    f"📈 IMPROVEMENT PATTERN: {pattern.improvement_template}" 
                    for pattern in relevant_patterns
                ])
        
        # Add inline comment guidance
        improvements.append("""
🔧 DEVELOPER FEEDBACK INTEGRATION:
- Pay attention to any inline comments like # FIX:, # TODO:, # IMPROVE:
- If code is rejected, learn from the feedback for future generations
- Focus on common issues: async/await, error handling, validation, performance
""")
        
        # Combine base prompt with learned improvements
        if improvements:
            adaptive_prompt = base_prompt + "\n\nLEARNED IMPROVEMENTS:\n" + "\n".join(improvements)
        else:
            adaptive_prompt = base_prompt
        
        return adaptive_prompt
    
    def _analyze_common_issues(self, feedback_list: List[FeedbackEntry]) -> List[str]:
        """Analyze common issues from feedback."""
        issues = []
        
        rejection_reasons = [f.rejection_reason for f in feedback_list if f.rejection_reason]
        inline_comments = [comment for f in feedback_list for comment in f.inline_comments]
        
        # Count common themes
        theme_counter = Counter()
        
        for reason in rejection_reasons:
            if reason:
                if 'async' in reason.lower():
                    theme_counter['async_issues'] += 1
                if 'error' in reason.lower() or 'exception' in reason.lower():
                    theme_counter['error_handling'] += 1
                if 'validation' in reason.lower():
                    theme_counter['validation_issues'] += 1
                if 'performance' in reason.lower():
                    theme_counter['performance_issues'] += 1
        
        for comment in inline_comments:
            if 'async' in comment.lower():
                theme_counter['async_issues'] += 1
            if 'error' in comment.lower():
                theme_counter['error_handling'] += 1
        
        # Convert to improvement suggestions
        for theme, count in theme_counter.most_common(3):
            if count >= 2:  # Only suggest if seen multiple times
                if theme == 'async_issues':
                    issues.append("Use async/await for database operations and API calls")
                elif theme == 'error_handling':
                    issues.append("Add comprehensive error handling with try/except blocks")
                elif theme == 'validation_issues':
                    issues.append("Include thorough input validation and data checks")
                elif theme == 'performance_issues':
                    issues.append("Optimize for performance with proper indexing and caching")
        
        return issues
    
    def _get_relevant_patterns(self, patterns: Dict[str, ImprovementPattern], file_type: str) -> List[ImprovementPattern]:
        """Get patterns relevant to the current file type."""
        relevant = []
        
        for pattern in patterns.values():
            if pattern.confidence_score > 0.7 and pattern.usage_count >= 2:
                # Check if pattern is relevant to file type
                if file_type == 'models' and any(kw in pattern.trigger_keywords for kw in ['model', 'field', 'relationship']):
                    relevant.append(pattern)
                elif file_type == 'views' and any(kw in pattern.trigger_keywords for kw in ['view', 'request', 'response']):
                    relevant.append(pattern)
                elif file_type == 'serializers' and any(kw in pattern.trigger_keywords for kw in ['serializer', 'validation']):
                    relevant.append(pattern)
        
        return relevant[:3]  # Limit to top 3 patterns

class FeedbackIntegratedAgent:
    """Agent that integrates developer feedback into the generation process."""
    
    def __init__(self, base_agent, feedback_storage: FeedbackStorage):
        self.base_agent = base_agent
        self.feedback_storage = feedback_storage
        self.pattern_learner = PatternLearner(feedback_storage)
        self.prompt_generator = AdaptivePromptGenerator(feedback_storage, self.pattern_learner)
        self.comment_parser = InlineCommentParser()
    
    async def generate_with_feedback_integration(
        self, 
        file_type: str, 
        erd: Dict, 
        context: Dict = None
    ) -> Tuple[str, Dict[str, Any]]:
        """Generate code with integrated feedback learning."""
        
        # Generate adaptive prompt based on learned patterns
        adaptive_prompt = self.prompt_generator.generate_adaptive_prompt(file_type, erd, context)
        
        # Generate code using the base agent
        generated_code = await self.base_agent.generate(adaptive_prompt)
        
        # Parse any inline comments in the generated code
        inline_comments = self.comment_parser.parse_comments(generated_code)
        
        # Prepare feedback metadata
        feedback_metadata = {
            'inline_comments': inline_comments,
            'adaptive_improvements': True,
            'learned_patterns_applied': len(self.pattern_learner.learned_patterns),
            'generation_timestamp': datetime.now().isoformat()
        }
        
        return generated_code, feedback_metadata
    
    def handle_user_feedback(
        self, 
        file_type: str, 
        original_code: str, 
        user_action: str,
        corrected_code: str = None,
        rejection_reason: str = None,
        satisfaction_score: float = None,
        erd_context: Dict = None
    ):
        """Handle and store user feedback for learning."""
        
        # Parse inline comments from original code
        inline_comments = self.comment_parser.parse_comments(original_code)
        all_comments = []
        for comment_type, comments in inline_comments.items():
            all_comments.extend(comments)
        
        # Determine satisfaction score based on action
        if satisfaction_score is None:
            if user_action == 'approve':
                satisfaction_score = 1.0
            elif user_action == 'reject':
                satisfaction_score = 0.0
            elif user_action == 'edit':
                satisfaction_score = 0.7  # Partial satisfaction
            else:
                satisfaction_score = 0.5
        
        # Extract tags from content
        tags = set()
        if 'async' in original_code.lower() or any('async' in comment for comment in all_comments):
            tags.add('async')
        if 'class' in original_code and 'models.Model' in original_code:
            tags.add('model')
        if 'ViewSet' in original_code:
            tags.add('viewset')
        if 'Serializer' in original_code:
            tags.add('serializer')
        
        # Create feedback entry
        feedback = FeedbackEntry(
            id="",  # Will be auto-generated
            timestamp=datetime.now(),
            file_type=file_type,
            original_code=original_code,
            corrected_code=corrected_code,
            inline_comments=all_comments,
            rejection_reason=rejection_reason,
            erd_context=erd_context,
            satisfaction_score=satisfaction_score,
            tags=tags
        )
        
        # Store feedback
        self.feedback_storage.store_feedback(feedback)
        
        # Re-analyze patterns for future use
        self.pattern_learner.analyze_feedback_patterns()
        
        print(f"📝 Feedback stored: {user_action} for {file_type} (satisfaction: {satisfaction_score:.1f})")
    
    def get_improvement_suggestions(self, code: str, file_type: str) -> List[str]:
        """Get improvement suggestions based on learned patterns."""
        inline_comments = self.comment_parser.parse_comments(code)
        suggestions = self.comment_parser.extract_improvement_suggestions(inline_comments)
        
        # Add pattern-based suggestions
        similar_feedback = self.feedback_storage.get_similar_feedback(code, file_type, limit=3)
        if similar_feedback:
            common_issues = self.prompt_generator._analyze_common_issues(similar_feedback)
            suggestions.extend(common_issues)
        
        return suggestions

class InteractiveFeedbackManager:
    """Interactive manager for handling developer feedback in real-time."""
    
    def __init__(self, feedback_agent: FeedbackIntegratedAgent):
        self.feedback_agent = feedback_agent
    
    def interactive_review(self, file_type: str, generated_code: str, erd_context: Dict = None) -> Dict[str, Any]:
        """Interactive review process with feedback collection."""
        print(f"\n{'='*60}")
        print(f"🔍 REVIEW: {file_type}")
        print(f"{'='*60}")
        
        # Show code preview
        lines = generated_code.split('\n')
        preview_lines = min(20, len(lines))
        print(f"\n📄 Generated Code Preview ({preview_lines}/{len(lines)} lines):")
        print("-" * 40)
        for i, line in enumerate(lines[:preview_lines], 1):
            print(f"{i:3d} | {line}")
        if len(lines) > preview_lines:
            print(f"... and {len(lines) - preview_lines} more lines")
        
        # Check for inline comments
        inline_comments = self.feedback_agent.comment_parser.parse_comments(generated_code)
        if inline_comments:
            print(f"\n💬 Inline Comments Detected:")
            for comment_type, comments in inline_comments.items():
                for comment in comments:
                    print(f"   • {comment_type.upper()}: {comment}")
        
        # Get improvement suggestions
        suggestions = self.feedback_agent.get_improvement_suggestions(generated_code, file_type)
        if suggestions:
            print(f"\n💡 Improvement Suggestions:")
            for i, suggestion in enumerate(suggestions[:3], 1):
                print(f"   {i}. {suggestion}")
        
        # Interactive feedback collection
        print(f"\n🎯 Actions:")
        print("   [a] Approve (use as-is)")
        print("   [r] Reject (regenerate)")
        print("   [e] Edit (make changes)")
        print("   [c] Add comment (inline feedback)")
        print("   [s] Skip (no feedback)")
        
        while True:
            choice = input("\nYour choice: ").lower().strip()
            
            if choice == 'a':
                self.feedback_agent.handle_user_feedback(
                    file_type, generated_code, 'approve', 
                    erd_context=erd_context
                )
                return {'action': 'approve', 'code': generated_code}
            
            elif choice == 'r':
                reason = input("Why reject? (optional): ").strip()
                self.feedback_agent.handle_user_feedback(
                    file_type, generated_code, 'reject',
                    rejection_reason=reason or "User rejected",
                    erd_context=erd_context
                )
                return {'action': 'reject', 'reason': reason}
            
            elif choice == 'e':
                print("\n📝 Edit the code (paste your corrected version, end with '###'):")
                corrected_lines = []
                while True:
                    try:
                        line = input()
                        if line.strip() == '###':
                            break
                        corrected_lines.append(line)
                    except (EOFError, KeyboardInterrupt):
                        break
                
                corrected_code = '\n'.join(corrected_lines)
                if corrected_code.strip():
                    self.feedback_agent.handle_user_feedback(
                        file_type, generated_code, 'edit',
                        corrected_code=corrected_code,
                        erd_context=erd_context
                    )
                    return {'action': 'edit', 'code': corrected_code}
                else:
                    print("❌ No corrected code provided")
            
            elif choice == 'c':
                comment = input("Add inline comment (e.g., 'FIX: make this async'): ").strip()
                if comment:
                    # Add comment to code
                    commented_code = f"# {comment}\n{generated_code}"
                    self.feedback_agent.handle_user_feedback(
                        file_type, commented_code, 'comment',
                        erd_context=erd_context
                    )
                    return {'action': 'comment', 'code': commented_code}
            
            elif choice == 's':
                return {'action': 'skip', 'code': generated_code}
            
            else:
                print("❌ Invalid choice. Please select a, r, e, c, or s.")

# Usage Example and Integration
class FeedbackEnabledBackendBuilder:
    """Backend builder with integrated feedback learning."""
    
    def __init__(self, base_builder, feedback_db_path: str = ".feedback_db.sqlite"):
        self.base_builder = base_builder
        self.feedback_storage = FeedbackStorage(feedback_db_path)
        self.feedback_agents = {}
        
        # Create feedback-enabled agents for each component
        for component in ['models', 'serializers', 'views', 'urls', 'settings']:
            if hasattr(base_builder, f'{component.rstrip("s")}_agent'):
                base_agent = getattr(base_builder, f'{component.rstrip("s")}_agent')
                self.feedback_agents[component] = FeedbackIntegratedAgent(
                    base_agent, self.feedback_storage
                )
    
    async def generate_with_feedback(self, erd: Dict, interactive: bool = True) -> Dict[str, str]:
        """Generate backend with feedback integration."""
        print("🧠 Starting feedback-integrated generation...")
        
        results = {}
        feedback_manager = InteractiveFeedbackManager(None)
        
        for component, agent in self.feedback_agents.items():
            print(f"\n🔄 Generating {component}...")
            
            # Generate with feedback integration
            code, metadata = await agent.generate_with_feedback_integration(
                component, erd
            )
            
            if interactive:
                # Interactive review
                feedback_manager.feedback_agent = agent
                review_result = feedback_manager.interactive_review(
                    component, code, erd_context=erd
                )
                
                if review_result['action'] == 'reject':
                    # Regenerate with learned feedback
                    print(f"🔄 Regenerating {component} with feedback...")
                    code, metadata = await agent.generate_with_feedback_integration(
                        component, erd
                    )
                elif review_result['action'] == 'edit':
                    code = review_result['code']
            
            results[component] = code
        
        return results

# Demo function
async def demo_feedback_integration():
    """Demonstrate the feedback integration system."""
    print("🔧 Developer Feedback Integration Demo")
    print("=" * 50)
    
    # Mock ERD for demo
    erd = {
        "User": {
            "name": "CharField",
            "email": "EmailField"
        },
        "Product": {
            "name": "CharField",
            "price": "DecimalField",
            "user": "ForeignKey:User"
        }
    }
    
    # Initialize feedback system
    feedback_storage = FeedbackStorage(":memory:")  # In-memory for demo
    
    # Simulate base agent (simplified)
    class MockAgent:
        async def generate(self, prompt):
            return """
# Generated Django models
from django.db import models

class User(models.Model):
    name = models.CharField(max_length=100)
    email = models.EmailField()
    
    def __str__(self):
        return self.name

class Product(models.Model):
    name = models.CharField(max_length=200)
    price = models.DecimalField(max_digits=10, decimal_places=2)
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    
    def __str__(self):
        return self.name
"""
    
    # Create feedback-integrated agent
    mock_agent = MockAgent()
    feedback_agent = FeedbackIntegratedAgent(mock_agent, feedback_storage)
    
    # Simulate feedback scenarios
    print("\n1. 📝 Generating code with feedback integration...")
    code, metadata = await feedback_agent.generate_with_feedback_integration('models', erd)
    print("✅ Code generated successfully")
    
    print("\n2. 🔧 Simulating user feedback...")
    
    # Simulate rejection with feedback
    feedback_agent.handle_user_feedback(
        'models', code, 'reject',
        rejection_reason="Missing async support and error handling"
    )
    
    # Simulate edit with inline comments
    commented_code = """
# FIX: make this async
# IMPROVE: add error handling
from django.db import models

class User(models.Model):
    name = models.CharField(max_length=100)
    email = models.EmailField()
    
    def __str__(self):
        return self.name
"""
    
    feedback_agent.handle_user_feedback(
        'models', commented_code, 'edit',
        corrected_code=commented_code
    )
    
    print("\n3. 🧠 Learning from feedback...")
    patterns = feedback_agent.pattern_learner.analyze_feedback_patterns()
    print(f"   • Learned {len(patterns)} improvement patterns")
    
    suggestions = feedback_agent.get_improvement_suggestions(code, 'models')
    print(f"   • Generated {len(suggestions)} improvement suggestions")
    
    print("\n4. 📈 Adaptive prompt generation...")
    adaptive_prompt = feedback_agent.prompt_generator.generate_adaptive_prompt('models', erd)
    print("   • Enhanced prompt with learned improvements")
    print("   • System will improve over time with more feedback")
    
    print("\n🎉 Feedback integration demo complete!")
    print("   • Inline comments are parsed and understood")
    print("   • User feedback is stored and learned from")
    print("   • Prompts adapt based on common issues")
    print("   • System continuously improves")

if __name__ == "__main__":
    asyncio.run(demo_feedback_integration()) 