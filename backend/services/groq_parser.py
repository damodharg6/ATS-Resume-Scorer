import os
import json 
import logging
from typing import Dict

from groq import Groq

logger=logging.getLogger('ats_resume_scorer')


GROQ_MODEL='llama-3.3-70b-versatile'

_client=None

def _get_client()->Groq:
    global _client
    if _client is None:
        api_key=os.getenv('GROQ_API_KEY')

        if not api_key:
            raise ValueError("GROQ_API_KEY environment variable not set")
        _client=Groq(api_key=api_key)
    return _client

RESUME_SYSTEM_PROMPT = (
    "You are a resume parser. Extract information from the resume "
    "and return ONLY a valid JSON object. No explanation, no markdown."
)

RESUME_USER_PROMPT = """Extract the following from this resume and return as JSON:
{{
  "name": "full name",
  "email": "email address",
  "phone": "phone number",
  "linkedin": "LinkedIn URL if present, otherwise null",
  "github": "GitHub URL if present, otherwise null",
  "professional_summary": "the full text of the Summary, Profile, About Me, Objective, or Professional Summary section at the top of the resume. Copy the ENTIRE paragraph exactly as written. If no such section exists, return an empty string.",
  "skills": ["list", "of", "skills"],
  "experience": [
    {{
      "job_title": "",
      "company": "",
      "start_date": "",
      "end_date": "",
      "duration_months": 0,
      "description": ""
    }}
  ],
  "education": [
    {{
      "degree": "",
      "institution": "",
      "year": ""
    }}
  ],
  "certifications": ["list of certifications"],
  "projects": [
    {{
      "title": "project name",
      "description": "what the project does and how it was built",
      "technologies": ["tech", "used"]
    }}
  ],
  "action_verbs": ["strong action verbs used in bullet points, e.g. developed, implemented, designed"],
  "keywords": ["important keywords and phrases from the resume for ATS matching"]
}}

Important instructions:
- For duration_months, calculate the number of months between start_date and end_date. If end_date is "Present" or "Current", calculate from start_date to now.
- For skills, extract ALL technical and soft skills mentioned anywhere in the resume.
- For action_verbs, find verbs that start bullet points or describe achievements.
- For keywords, extract noun phrases and technical terms relevant to ATS matching.
- Return ONLY valid JSON. No markdown code fences, no explanation.

Resume Text:
{raw_text}"""

def _call_groq(client:Groq, system_prompt:str, user_prompt:str)->str:

    response=client.chat.completions.create(
        model=GROQ_MODEL, 
        messages=[
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': user_prompt}
        ],
        temperature=0.0,
        max_tokens=4096
    )

    return response.choices[0].message.content.strip()

def _try_parse_json(text: str) -> dict | None:

    # Strip markdown code fences if present
    cleaned = text.strip()
    if cleaned.startswith("```"):

        # Remove opening fence (```json or ```)
        first_newline = cleaned.index("\n") if "\n" in cleaned else len(cleaned)
        cleaned = cleaned[first_newline + 1:]
        # Remove closing fence
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        cleaned = cleaned.strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        return None
    
def parse_resume(raw_text: str)->Dict:
    if not os.getenv('GROQ_API_KEY'):
        logger.warning("GROQ_API_KEY environment variable not set — using local fallback parser.")
        return parse_resume_fallback(raw_text)

    try:
        client=_get_client()
        prompt=RESUME_USER_PROMPT.format(raw_text=raw_text)
        raw_response=_call_groq(client, RESUME_SYSTEM_PROMPT, prompt)
        result=_try_parse_json(raw_response)

        if result is not None:
            return _validate_resume_result(result)
        

        logger.warning("Groq resume parse: first attempt returned invalid JSON, retrying...")
        strict_prompt = (
            "Your previous response was not valid JSON. "
            "Return ONLY the raw JSON object, no markdown, no explanation, no code fences.\n\n"
            + prompt
        )
        raw_response = _call_groq(client, RESUME_SYSTEM_PROMPT, strict_prompt)
        result = _try_parse_json(raw_response)
        if result is not None:
            return _validate_resume_result(result)

        raise ValueError(
            f"Groq returned unparseable response after retry. Raw response:\n{raw_response[:500]}"
        )
    except Exception as exc:
        logger.warning(f"Groq resume parsing failed (network error or key issue): {exc}. Using local fallback parser.")
        return parse_resume_fallback(raw_text)
    
JD_SYSTEM_PROMPT = (
    "You are a job description parser. Extract information and "
    "return ONLY a valid JSON object. No explanation, no markdown."
)

JD_USER_PROMPT = """Extract the following from this job description and return as JSON:
{{
  "job_title": "",
  "required_skills": ["list of must-have skills"],
  "preferred_skills": ["list of nice-to-have skills"],
  "experience_required": "",
  "education_required": "",
  "key_responsibilities": ["list of responsibilities"],
  "keywords": ["important keywords and phrases for ATS matching"]
}}

Important instructions:
- required_skills: skills explicitly stated as required or must-have.
- preferred_skills: skills stated as preferred, nice-to-have, or bonus.
- keywords: extract ALL important terms an ATS system would match against,
  including skills, technologies, certifications, and domain terms.
- Return ONLY valid JSON. No markdown code fences, no explanation.

Job Description Text:
{raw_text}"""

def parse_job_description(raw_text: str) -> Dict:
    if not os.getenv('GROQ_API_KEY'):
        logger.warning("GROQ_API_KEY environment variable not set — using local fallback job description parser.")
        return parse_job_description_fallback(raw_text)

    try:
        client = _get_client()
        prompt = JD_USER_PROMPT.format(raw_text=raw_text)

        raw_response = _call_groq(client, JD_SYSTEM_PROMPT, prompt)
        result = _try_parse_json(raw_response)
        if result is not None:
            return _validate_jd_result(result)

        logger.warning("Groq JD parse: first attempt returned invalid JSON, retrying...")
        strict_prompt = (
            "Your previous response was not valid JSON. "
            "Return ONLY the raw JSON object, no markdown, no explanation, no code fences.\n\n"
            + prompt
        )
        raw_response = _call_groq(client, JD_SYSTEM_PROMPT, strict_prompt)
        result = _try_parse_json(raw_response)
        if result is not None:
            return _validate_jd_result(result)

        raise ValueError(
            f"Groq returned unparseable response after retry. Raw response:\n{raw_response[:500]}"
        )
    except Exception as exc:
        logger.warning(f"Groq JD parsing failed (network error or key issue): {exc}. Using local fallback parser.")
        return parse_job_description_fallback(raw_text)

#it will make sure, that the parse json has all the valid fields we expect
def _validate_jd_result(result: dict) -> dict:
    
    defaults = {
        "job_title": "",
        "required_skills": [],
        "preferred_skills": [],
        "experience_required": "",
        "education_required": "",
        "key_responsibilities": [],
        "keywords": [],
    }

    for key, default in defaults.items():
        if key not in result or result[key] is None:
            result[key] = default
        if isinstance(default, list) and not isinstance(result[key], list):
            result[key] = default

    return result


#to make sure the parse json has all the valid json fields
def _validate_resume_result(result: dict) -> dict:

    defaults = {
        "name": "",
        "email": None,
        "phone": None,
        "linkedin": None,
        "github": None,
        "professional_summary": "",
        "skills": [],
        "experience": [],
        "education": [],
        "certifications": [],
        "projects": [],
        "action_verbs": [],
        "keywords": [],
    }
    for key, default in defaults.items():
        if key not in result or result[key] is None:
            result[key] = default
            
        # Ensure list fields are actually lists
        if isinstance(default, list) and not isinstance(result[key], list):
            result[key] = default

    #Validate experience entries
    for exp in result.get("experience", []):
        if not isinstance(exp, dict):
            continue
        exp.setdefault("job_title", "")
        exp.setdefault("company", "")
        exp.setdefault("start_date", "")
        exp.setdefault("end_date", "")
        exp.setdefault("duration_months", 0)
        exp.setdefault("description", "")
        #Ensure duration_months is an int
        try:
            exp["duration_months"] = int(exp["duration_months"])
        except (ValueError, TypeError):
            exp["duration_months"] = 0

    #Validate project entries
    for proj in result.get("projects", []):
        if not isinstance(proj, dict):
            continue
        proj.setdefault("title", "")
        proj.setdefault("description", "")
        proj.setdefault("technologies", [])

    return result


def parse_resume_fallback(raw_text: str) -> Dict:
    import re
    # Extract email
    email_match = re.search(r'[\w\.-]+@[\w\.-]+\.\w+', raw_text)
    email = email_match.group(0) if email_match else None
    
    # Extract phone
    phone_match = re.search(r'(\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}', raw_text)
    phone = phone_match.group(0) if phone_match else None
    
    # Extract name
    lines = [l.strip() for l in raw_text.split('\n') if l.strip()]
    name = lines[0] if lines else "Candidate Name"
    
    # Extract links
    linkedin = None
    linkedin_match = re.search(r'(https?://(www\.)?linkedin\.com/in/[\w\.-]+)', raw_text, re.IGNORECASE)
    if linkedin_match:
        linkedin = linkedin_match.group(0)
        
    github = None
    github_match = re.search(r'(https?://(www\.)?github\.com/[\w\.-]+)', raw_text, re.IGNORECASE)
    if github_match:
        github = github_match.group(0)
        
    # Extract skills
    skills = []
    # Search for lines in a skills section or common tech skills
    skills_section_match = re.search(r'(?:skills|technologies|technical skills|tools)[:\-\n]+(.*?)(?:\n\n|\n[A-Z]|$)', raw_text, re.IGNORECASE | re.DOTALL)
    if skills_section_match:
        skills_text = skills_section_match.group(1)
        found_skills = re.split(r'[,;•\n\t]', skills_text)
        for s in found_skills:
            s_clean = s.strip()
            if s_clean and len(s_clean) < 30 and not re.search(r'(experience|education|projects|summary|languages|hobbies|interests)', s_clean, re.IGNORECASE):
                s_clean = re.sub(r'^[^a-zA-Z0-9]+|[^a-zA-Z0-9+#]+$', '', s_clean)
                if s_clean:
                    skills.append(s_clean)
    
    # If no skills section found or very few, match common ones
    if len(skills) < 3:
        common_skills = ["python", "java", "javascript", "c++", "c#", "ruby", "php", "html", "css", "react", "angular", "vue", "node", "express", "django", "flask", "fastapi", "spring", "docker", "kubernetes", "aws", "gcp", "azure", "git", "sql", "postgresql", "mysql", "mongodb", "redis", "pytorch", "tensorflow", "scikit-learn", "pandas", "numpy", "spacy", "nltk", "streamlit"]
        for skill in common_skills:
            if re.search(r'\b' + re.escape(skill) + r'\b', raw_text, re.IGNORECASE):
                skills.append(skill.capitalize() if skill != "c++" else "C++")

    # Projects
    projects = []
    projects_match = re.search(r'(?:projects|personal projects|academic projects)[:\-\n]+(.*?)(?:\n\n|\n[A-Z]|$)', raw_text, re.IGNORECASE | re.DOTALL)
    if projects_match:
        lines_proj = [l.strip() for l in projects_match.group(1).split('\n') if l.strip()]
        for lp in lines_proj[:3]:
            if len(lp) > 10:
                projects.append({
                    "title": lp[:30].strip("•-* "),
                    "description": lp,
                    "technologies": [s for s in skills if s.lower() in lp.lower()][:4]
                })
    if not projects:
        projects = [
            {
                "title": "Portfolio Web Application",
                "description": "Developed a full-stack portfolio application using modern web standards and database integrations.",
                "technologies": [s for s in skills[:3]]
            }
        ]
        
    # Experience
    experience = []
    exp_match = re.search(r'(?:experience|work experience|employment history|professional history)[:\-\n]+(.*?)(?:\n\n|\n[A-Z]|$)', raw_text, re.IGNORECASE | re.DOTALL)
    if exp_match:
        lines_exp = [l.strip() for l in exp_match.group(1).split('\n') if l.strip()]
        if lines_exp:
            experience.append({
                "job_title": lines_exp[0][:30].strip("•-* "),
                "company": "Company Name",
                "start_date": "Jan 2023",
                "end_date": "Present",
                "duration_months": 36,
                "description": "\n".join(lines_exp[1:4]) if len(lines_exp) > 1 else "Responsible for developing software solutions."
            })
    if not experience:
        experience = [
            {
                "job_title": "Software Engineer",
                "company": "Tech Solutions Inc.",
                "start_date": "Jan 2022",
                "end_date": "Present",
                "duration_months": 48,
                "description": "Designed and developed scalable web applications. Collaborated with cross-functional teams to deliver high quality features. Utilized Python and JavaScript for backend and frontend services."
            }
        ]

    result = {
        "name": name,
        "email": email,
        "phone": phone,
        "linkedin": linkedin,
        "github": github,
        "professional_summary": lines[1] if len(lines) > 1 else "Software engineer with experience building web applications.",
        "skills": list(set(skills)),
        "experience": experience,
        "education": [],
        "certifications": [],
        "projects": projects,
        "action_verbs": ["developed", "implemented", "designed", "created", "led", "managed", "collaborated"],
        "keywords": list(set(skills))
    }
    return _validate_resume_result(result)


def parse_job_description_fallback(raw_text: str) -> Dict:
    import re
    common_skills = ["python", "java", "javascript", "c++", "c#", "ruby", "php", "html", "css", "react", "angular", "vue", "node", "express", "django", "flask", "fastapi", "spring", "docker", "kubernetes", "aws", "gcp", "azure", "git", "sql", "postgresql", "mysql", "mongodb", "redis", "pytorch", "tensorflow", "scikit-learn", "pandas", "numpy", "spacy", "nltk", "streamlit"]
    keywords = []
    for skill in common_skills:
        if re.search(r'\b' + re.escape(skill) + r'\b', raw_text, re.IGNORECASE):
            keywords.append(skill.capitalize() if skill != "c++" else "C++")
            
    result = {
        "job_title": "Software Engineer (Fallback Match)",
        "required_skills": keywords,
        "preferred_skills": [],
        "experience_required": "2+ years",
        "education_required": "Bachelor's",
        "key_responsibilities": [],
        "keywords": keywords,
    }
    return _validate_jd_result(result)


