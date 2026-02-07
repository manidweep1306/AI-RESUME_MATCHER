import re

class ResumeExplainer:
    def explain(self, resume_text: str, job_text: str):
        resume_words = set(re.findall(r"\w+", resume_text.lower()))
        job_words = set(re.findall(r"\w+", job_text.lower()))

        common = resume_words.intersection(job_words)

        top_keywords = list(common)[:10]

        if not top_keywords:
            return "No strong keyword overlap found."

        return f"Matched based on overlapping skills/keywords: {', '.join(top_keywords)}"
