"""Skill normalisation against a standard taxonomy (design §2.2 step 4).

Maps surface forms to canonical names so 'React.js', 'ReactJS', and 'react'
all become 'React'. This runs before scoring: matching raw strings would
under-count candidates who happen to spell a skill differently from the job
description.
"""

from __future__ import annotations

import re

# canonical name -> alternative spellings
SKILL_ALIASES: dict[str, list[str]] = {
    # --- Languages ---
    "Python": ["python3", "py", "python 3"],
    "JavaScript": ["js", "ecmascript", "es6", "es2015", "vanilla js"],
    "TypeScript": ["ts"],
    "Java": ["java se", "core java", "java ee", "j2ee"],
    "C#": ["c sharp", "csharp", "dotnet c#"],
    "C++": ["cpp", "c plus plus"],
    "Go": ["golang"],
    "Ruby": ["ruby lang"],
    "PHP": [],
    "Swift": [],
    "Kotlin": [],
    "Rust": ["rust lang"],
    "Scala": [],
    "R": ["r language"],
    "SQL": ["structured query language"],
    "Shell": ["bash", "shell scripting", "zsh", "sh"],
    # --- Frontend ---
    "React": ["react.js", "reactjs", "react js", "react native"],
    "Angular": ["angular.js", "angularjs", "angular 2+"],
    "Vue.js": ["vue", "vuejs", "vue 3"],
    "Next.js": ["nextjs", "next js"],
    "Svelte": ["sveltekit"],
    "HTML": ["html5"],
    "CSS": ["css3"],
    "Tailwind CSS": ["tailwind", "tailwindcss"],
    "Redux": ["redux toolkit"],
    # --- Backend / frameworks ---
    "Node.js": ["node", "nodejs", "node js"],
    "Django": ["django rest framework", "drf"],
    "Flask": [],
    "FastAPI": ["fast api"],
    "Spring Boot": ["spring", "springboot", "spring framework"],
    "Express.js": ["express", "expressjs"],
    "Rails": ["ruby on rails", "ror"],
    "ASP.NET": ["asp net", "aspnet", ".net core", "dotnet core", ".net"],
    "GraphQL": ["graph ql"],
    "REST APIs": ["rest", "restful", "restful apis", "rest api"],
    "gRPC": ["grpc"],
    # --- Data / databases ---
    "PostgreSQL": ["postgres", "psql", "postgre sql"],
    "MySQL": ["my sql", "mariadb"],
    "MongoDB": ["mongo"],
    "Redis": [],
    "Elasticsearch": ["elastic search", "elk", "opensearch"],
    "Cassandra": ["apache cassandra"],
    "DynamoDB": ["dynamo db"],
    "Snowflake": [],
    "Apache Kafka": ["kafka"],
    "Apache Spark": ["spark", "pyspark"],
    "Airflow": ["apache airflow"],
    "dbt": ["data build tool"],
    "ETL": ["etl pipelines", "elt"],
    # --- Cloud / infra ---
    "AWS": ["amazon web services", "ec2", "amazon aws"],
    "Azure": ["microsoft azure", "ms azure"],
    "GCP": ["google cloud", "google cloud platform"],
    "Docker": ["containerization", "containers"],
    "Kubernetes": ["k8s", "kubernetes cluster", "eks", "gke", "aks"],
    "Terraform": ["hashicorp terraform", "iac terraform"],
    "Ansible": [],
    "Jenkins": [],
    "CI/CD": ["cicd", "ci cd", "continuous integration", "continuous delivery"],
    "GitHub Actions": ["gh actions", "github action"],
    "Linux": ["unix", "ubuntu", "centos", "rhel"],
    "Nginx": [],
    "Prometheus": [],
    "Grafana": [],
    # --- ML / AI ---
    "Machine Learning": ["ml", "machine-learning"],
    "Deep Learning": ["dl", "neural networks"],
    "TensorFlow": ["tensor flow", "tf"],
    "PyTorch": ["torch", "py torch"],
    "scikit-learn": ["sklearn", "scikit learn"],
    "NLP": ["natural language processing"],
    "Computer Vision": ["cv", "opencv", "image processing"],
    "LLMs": ["large language models", "llm", "genai", "generative ai"],
    "Pandas": ["pandas library"],
    "NumPy": ["numpy library"],
    # --- Tools / practices ---
    "Git": ["version control", "github", "gitlab", "bitbucket"],
    "Agile": ["scrum", "kanban", "agile methodology"],
    "Jira": ["atlassian jira"],
    "Microservices": ["micro services", "microservice architecture"],
    "System Design": ["distributed systems", "software architecture"],
    "TDD": ["test driven development", "test-driven development"],
    "Unit Testing": ["unit tests", "pytest", "junit", "jest"],
    # --- Non-engineering ---
    "Project Management": ["pm", "program management"],
    "Product Management": ["product owner", "product strategy"],
    "Digital Marketing": ["online marketing", "performance marketing"],
    "SEO": ["search engine optimization"],
    "Salesforce": ["sfdc"],
    "Excel": ["microsoft excel", "ms excel", "advanced excel"],
    "Tableau": [],
    "Power BI": ["powerbi", "power-bi"],
    "Figma": [],
    "UI/UX Design": ["ux design", "ui design", "user experience", "product design"],
    "Financial Modeling": ["financial modelling"],
    "Accounting": ["bookkeeping"],
    "Recruitment": ["talent acquisition", "hiring", "sourcing"],
    "Customer Success": ["customer support", "client success"],
    "Communication": ["communication skills", "verbal communication"],
    "Leadership": ["team leadership", "people management"],
}

# Reverse lookup: normalised surface form -> canonical name. Built once.
_LOOKUP: dict[str, str] = {}
for _canonical, _aliases in SKILL_ALIASES.items():
    _LOOKUP[_canonical.lower()] = _canonical
    for _alias in _aliases:
        _LOOKUP[_alias.lower()] = _canonical

_PUNCT = re.compile(r"[^\w+#./\- ]+")
_WS = re.compile(r"\s+")

PROFICIENCY_LEVELS = ("beginner", "intermediate", "advanced", "expert")

# A proficiency or duration qualifier trailing a skill name.
_QUALIFIER_SUFFIX = re.compile(
    r"(?<=\w)\s*[-–(]?\s*"
    r"(?:\d+\+?\s*(?:years?|yrs?)|beginner|intermediate|advanced|expert)"
    r"\s*\)?\s*$"
)


def _normalise(raw: str) -> str:
    text = _PUNCT.sub(" ", raw.lower())
    return _WS.sub(" ", text).strip()


def canonicalize(skill: str) -> str:
    """Map one skill string to its canonical name.

    Unknown skills are title-cased and returned as-is: the taxonomy is a
    normaliser, not a whitelist, so a niche skill is still usable for matching.
    """
    if not skill or not skill.strip():
        return ""

    normalized = _normalise(skill)
    if not normalized:
        return ""
    if normalized in _LOOKUP:
        return _LOOKUP[normalized]

    # Strip trailing qualifiers such as "Python (advanced)" or "React - 3 yrs".
    # Note this runs on the *normalised* form, where brackets have already been
    # flattened to spaces, so the separator has to be optional. The lookbehind
    # and the end anchor keep it from eating a skill that merely starts with a
    # qualifier word ("Advanced Analytics").
    stripped = _QUALIFIER_SUFFIX.sub("", normalized).strip()
    if stripped and stripped in _LOOKUP:
        return _LOOKUP[stripped]

    # Preserve intentional casing for known-style tokens (C#, C++, .NET).
    cleaned = skill.strip()
    if cleaned.isupper() and len(cleaned) <= 6:
        return cleaned
    return " ".join(w if w.isupper() else w.capitalize() for w in cleaned.split())


def normalize_skills(skills: list | None) -> list[dict]:
    """Canonicalise a mixed list of skill strings/dicts.

    Accepts ``["python", {"name": "React.js", "proficiency": "advanced"}]`` and
    returns a deduplicated list of ``{"name", "proficiency", "years"}`` dicts,
    keeping the richest information seen for each canonical skill.
    """
    out: dict[str, dict] = {}
    for entry in skills or []:
        if isinstance(entry, str):
            name, proficiency, years = entry, None, None
        elif isinstance(entry, dict):
            name = str(entry.get("name") or entry.get("skill") or "")
            proficiency = entry.get("proficiency") or entry.get("level")
            years = entry.get("years") or entry.get("years_of_experience")
        else:
            continue

        canonical = canonicalize(name)
        if not canonical:
            continue

        if isinstance(proficiency, str):
            proficiency = proficiency.strip().lower()
            if proficiency not in PROFICIENCY_LEVELS:
                proficiency = None
        else:
            proficiency = None

        try:
            years = float(years) if years is not None else None
        except (TypeError, ValueError):
            years = None

        existing = out.get(canonical)
        if existing is None:
            out[canonical] = {
                "name": canonical,
                "proficiency": proficiency,
                "years": years,
            }
        else:
            # Merge: keep whichever entry carries more detail.
            if existing["proficiency"] is None and proficiency:
                existing["proficiency"] = proficiency
            if years is not None and (
                existing["years"] is None or years > existing["years"]
            ):
                existing["years"] = years

    return list(out.values())


def skill_names(skills: list) -> list[str]:
    """Canonical names only, deduplicated and order-preserving."""
    return [s["name"] for s in normalize_skills(skills)]


def match_skills(
    candidate_skills: list, required_skills: list
) -> tuple[list[str], list[str]]:
    """Split required skills into (matched, missing) after canonicalisation."""
    have = {s.lower() for s in skill_names(candidate_skills)}
    matched: list[str] = []
    missing: list[str] = []
    for name in skill_names(required_skills):
        (matched if name.lower() in have else missing).append(name)
    return matched, missing
