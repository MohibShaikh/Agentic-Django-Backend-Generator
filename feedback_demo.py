#!/usr/bin/env python3
"""
Simple Demo of Developer Feedback Integration
=============================================

🔧 SHOWS HOW FEEDBACK INTEGRATION SOLVES THE PROBLEMS:

Problem ❌ → Solution ✅
- User input ignored once code is generated → Inline comment parsing and interactive review
- No learning from corrections → Feedback storage and pattern learning
"""

import re
import sqlite3
from datetime import datetime
from dataclasses import dataclass
from typing import List, Dict

@dataclass
class InlineComment:
    type: str
    content: str
    line_number: int

class FeedbackDemo:
    """Simple demo of feedback integration capabilities."""
    
    def __init__(self):
        self.feedback_db = ":memory:"  # In-memory for demo
        self._init_demo_db()
    
    def _init_demo_db(self):
        """Initialize demo database."""
        conn = sqlite3.connect(self.feedback_db)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE feedback (
                id INTEGER PRIMARY KEY,
                timestamp TEXT,
                file_type TEXT,
                action TEXT,
                feedback_text TEXT,
                satisfaction INTEGER
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE patterns (
                pattern TEXT PRIMARY KEY,
                count INTEGER DEFAULT 1
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def parse_inline_comments(self, code: str) -> List[InlineComment]:
        """Parse inline developer comments."""
        comments = []
        lines = code.split('\n')
        
        patterns = {
            'FIX': r'#\s*FIX[:\s]+(.*?)$',
            'TODO': r'#\s*TODO[:\s]+(.*?)$',
            'IMPROVE': r'#\s*IMPROVE[:\s]+(.*?)$',
            'BUG': r'#\s*BUG[:\s]+(.*?)$',
            'ASYNC': r'#\s*ASYNC[:\s]+(.*?)$'
        }
        
        for line_num, line in enumerate(lines, 1):
            for comment_type, pattern in patterns.items():
                match = re.search(pattern, line, re.IGNORECASE)
                if match:
                    comments.append(InlineComment(
                        type=comment_type,
                        content=match.group(1).strip(),
                        line_number=line_num
                    ))
        
        return comments
    
    def store_feedback(self, file_type: str, action: str, feedback_text: str = "", satisfaction: int = 3):
        """Store developer feedback."""
        conn = sqlite3.connect(self.feedback_db)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO feedback (timestamp, file_type, action, feedback_text, satisfaction)
            VALUES (?, ?, ?, ?, ?)
        ''', (
            datetime.now().isoformat(),
            file_type,
            action,
            feedback_text,
            satisfaction
        ))
        
        conn.commit()
        conn.close()
        
        print(f"📝 Feedback stored: {action} for {file_type} (satisfaction: {satisfaction}/5)")
    
    def learn_pattern(self, pattern: str):
        """Learn a pattern from feedback."""
        conn = sqlite3.connect(self.feedback_db)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO patterns (pattern, count)
            VALUES (?, COALESCE((SELECT count FROM patterns WHERE pattern = ?) + 1, 1))
        ''', (pattern, pattern))
        
        conn.commit()
        conn.close()
        
        print(f"🧠 Learned pattern: {pattern}")
    
    def get_learned_patterns(self) -> List[tuple]:
        """Get learned patterns."""
        conn = sqlite3.connect(self.feedback_db)
        cursor = conn.cursor()
        
        cursor.execute('SELECT pattern, count FROM patterns ORDER BY count DESC')
        patterns = cursor.fetchall()
        
        conn.close()
        return patterns

def demo_inline_comment_parsing():
    """Demo 1: Inline Comment Parsing"""
    print("\n🔧 DEMO 1: INLINE COMMENT PARSING")
    print("=" * 45)
    
    demo = FeedbackDemo()
    
    # Sample code with inline comments
    code_with_comments = """
# FIX: make this async
# TODO: add validation
from django.db import models

class User(models.Model):
    name = models.CharField(max_length=100)
    # BUG: email should be unique
    email = models.EmailField()
    
    # IMPROVE: optimize this query
    def get_profile(self):
        return self.profile
    
    # ASYNC: make this async
    def send_email(self):
        pass
"""
    
    print("📄 Sample Code with Inline Comments:")
    print("-" * 30)
    for i, line in enumerate(code_with_comments.strip().split('\n'), 1):
        marker = "➤" if line.strip().startswith('#') and any(x in line for x in ['FIX', 'TODO', 'BUG', 'IMPROVE', 'ASYNC']) else " "
        print(f"{marker} {i:2d} | {line}")
    
    # Parse comments
    comments = demo.parse_inline_comments(code_with_comments)
    
    print(f"\n💬 Parsed {len(comments)} Inline Comments:")
    for comment in comments:
        print(f"   Line {comment.line_number}: {comment.type} - {comment.content}")
    
    print("\n✅ RESULT: System can now understand developer feedback embedded in code!")

def demo_feedback_learning():
    """Demo 2: Learning from Developer Feedback"""
    print("\n🧠 DEMO 2: LEARNING FROM DEVELOPER FEEDBACK")
    print("=" * 50)
    
    demo = FeedbackDemo()
    
    # Simulate developer feedback over time
    feedback_scenarios = [
        ("models", "reject", "Missing async support", 2),
        ("views", "edit", "Need error handling", 3),
        ("models", "reject", "No async/await", 1),
        ("serializers", "approve", "Looks good", 5),
        ("views", "reject", "Missing try/except blocks", 2),
        ("models", "edit", "Add async def", 4),
    ]
    
    print("📝 Simulating Developer Feedback Over Time:")
    for i, (file_type, action, feedback, satisfaction) in enumerate(feedback_scenarios, 1):
        print(f"\n   Feedback {i}:")
        print(f"   • File: {file_type}")
        print(f"   • Action: {action}")
        print(f"   • Comment: {feedback}")
        print(f"   • Satisfaction: {satisfaction}/5")
        
        demo.store_feedback(file_type, action, feedback, satisfaction)
        
        # Learn patterns from feedback
        if "async" in feedback.lower():
            demo.learn_pattern("needs_async")
        if "error" in feedback.lower() or "try" in feedback.lower():
            demo.learn_pattern("needs_error_handling")
        if action == "reject":
            demo.learn_pattern("quality_issues")
    
    # Show learned patterns
    patterns = demo.get_learned_patterns()
    print(f"\n🎯 System Learned {len(patterns)} Patterns:")
    for pattern, count in patterns:
        print(f"   • {pattern}: seen {count} times")
    
    print("\n✅ RESULT: System learns from developer feedback and improves over time!")

def demo_adaptive_prompts():
    """Demo 3: Adaptive Prompt Generation"""
    print("\n📈 DEMO 3: ADAPTIVE PROMPT GENERATION")
    print("=" * 45)
    
    # Original prompt
    original_prompt = """
Generate Django models with the following requirements:
- Include all necessary imports
- Add proper field types and relationships
- Follow Django best practices
"""
    
    # Learned improvements from feedback
    learned_improvements = [
        "Use async/await for database operations (seen 3 times)",
        "Add try/except blocks for error handling (seen 2 times)",
        "Include proper validation methods (seen 1 time)"
    ]
    
    print("📄 Original Prompt:")
    print(original_prompt.strip())
    
    print("\n🧠 + Learned Improvements:")
    for improvement in learned_improvements:
        print(f"   • {improvement}")
    
    # Enhanced prompt
    enhanced_prompt = original_prompt + "\nIMPORTANT - Based on developer feedback:\n"
    for i, improvement in enumerate(learned_improvements, 1):
        enhanced_prompt += f"{i}. {improvement}\n"
    enhanced_prompt += "\nPlease incorporate these learnings into your code generation."
    
    print("\n📈 Enhanced Adaptive Prompt:")
    print("-" * 30)
    print(enhanced_prompt.strip())
    
    print("\n✅ RESULT: Prompts adapt based on developer feedback patterns!")

def demo_interactive_review():
    """Demo 4: Interactive Review Process"""
    print("\n👥 DEMO 4: INTERACTIVE REVIEW PROCESS")
    print("=" * 45)
    
    sample_code = """from django.db import models

class User(models.Model):
    name = models.CharField(max_length=100)
    email = models.EmailField()
    
    def __str__(self):
        return self.name
"""
    
    print("📄 Generated Code:")
    for i, line in enumerate(sample_code.strip().split('\n'), 1):
        print(f"  {i:2d} | {line}")
    
    print("\n🎯 Interactive Review Options:")
    print("   [1] ✅ Approve (code looks good)")
    print("   [2] ❌ Reject (regenerate)")  
    print("   [3] ✏️  Edit (make changes)")
    print("   [4] 💬 Add feedback (comment)")
    print("   [5] ⏭️  Skip (no action)")
    
    print("\n💡 Based on User Choice:")
    print("   • Approve → Learns this pattern is good")
    print("   • Reject → Learns to avoid similar patterns")
    print("   • Edit → Learns from the corrections")
    print("   • Comment → Stores feedback for future use")
    
    print("\n✅ RESULT: Interactive workflow captures all developer input!")

def main():
    """Run all feedback integration demos."""
    print("🔧 DEVELOPER FEEDBACK INTEGRATION DEMO")
    print("=" * 55)
    print("Solving the problems:")
    print("❌ User input ignored once code is generated")
    print("❌ No learning from corrections")
    print("↓")
    print("✅ Inline comment parsing + Interactive review")
    print("✅ Feedback storage + Pattern learning")
    
    # Run demos
    demo_inline_comment_parsing()
    demo_feedback_learning() 
    demo_adaptive_prompts()
    demo_interactive_review()
    
    print("\n🎉 FEEDBACK INTEGRATION COMPLETE!")
    print("=" * 40)
    print("✅ Problems Solved:")
    print("   • Inline comments are parsed and understood")
    print("   • Interactive review captures all feedback")
    print("   • System learns from corrections and rejections")
    print("   • Prompts adapt based on learned patterns")
    print("   • Continuous improvement with each interaction")
    
    print("\n💻 Implementation:")
    print("   • Replace standard builder with FeedbackIntegratedBuilder")
    print("   • Enable interactive mode for review workflow")
    print("   • System automatically improves with usage!")

if __name__ == "__main__":
    main() 