#topic_classifier.py 
from transformers import pipeline
import json
import torch

class TopicClassifier:
    def __init__(self):
        self.categories = self._get_categories()
        device = 0 if torch.cuda.is_available() else -1  # Use GPU if available
        self.classifier = pipeline(
            "zero-shot-classification",
            model="MoritzLaurer/deberta-v3-large-zeroshot-v2.0",
            device=device
        )

    def _get_categories(self):
        return [
            "Definition", "Explanation", "Purpose / Function", "Significance / Importance", "Scope & Applications",
            "Use Cases", "Features", "Benefits", "Challenges", "Limitations", "Advantages", "Disadvantages",
            "Comparison / Contrast", "Trends / Predictions", "Cause & Effect", "Problem-Solution", "Evaluation / Judgment",
            "Hypothesis / Assumption", "Insights", "Critical Review", "Process / Steps", "Methodology / Approach",
            "Algorithm / Computation", "Implementation", "Framework", "Workflow", "Blueprint", "Strategy",
            "Model Architecture", "Research & Experimentation", "Scientific Principle", "Engineering / Design",
            "Statistics / Research Findings", "Formula / Calculation", "Hypothesis Testing", "Simulation", "Technical Details",
            "Protocols", "Experiments", "Validation", "Testing & Debugging", "Historical Background", "Milestones & Discoveries",
            "Future Scope & Innovation", "Ethical Concerns", "Legal & Compliance", "Philosophy / Theory", "Policy",
            "Curriculum Content", "Education & Learning", "Training Modules", "Illustration", "Analogy", "Case Study",
            "Example", "Infographic", "Quote", "Diagram", "Visualization", "Demonstration", "Interactive Example",
            "Business Strategy", "Marketing & Branding", "Finance & Economics", "Risk Management", "User Experience (UX)",
            "Operations", "Market Analysis", "Product Lifecycle", "Revenue Model", "Sales Funnel", "Customer Journey",
            "Client Requirements", "Artificial Intelligence", "Machine Learning", "Blockchain", "Internet of Things (IoT)",
            "Cybersecurity", "Cloud Computing", "Edge Computing", "Data Analytics", "Big Data", "DevOps", "Mobile Apps",
            "Web Development", "API Integration", "Software Testing", "Version Control", "Automation", "Digital Transformation",
            "Healthcare & Medicine", "Education", "Agriculture", "Retail & eCommerce", "Banking & Finance",
            "Telecommunications", "Smart Cities", "Supply Chain", "Manufacturing", "Environmental & Sustainability"
        ]

    def classify(self, text):
        result = self.classifier(text, candidate_labels=self.categories, multi_label=False)
        return result['labels'][0]
