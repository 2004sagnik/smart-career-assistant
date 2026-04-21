import os
import chromadb
from sentence_transformers import SentenceTransformer
from typing import Tuple, List

# ── Knowledge Base Documents ─────────────────────────────────────────────────

DOCUMENTS = [
    {
        "id": "doc_001",
        "topic": "SSC CHSL Exam Pattern",
        "text": (
            "SSC CHSL (Combined Higher Secondary Level) is conducted by the Staff Selection Commission "
            "for posts like LDC, JSA, PA, SA, and DEO. The exam has two tiers. "
            "Tier-1 is a Computer Based Examination with 100 questions worth 200 marks in 60 minutes. "
            "It covers English Language (25 questions, 50 marks), General Intelligence (25 questions, 50 marks), "
            "Quantitative Aptitude (25 questions, 50 marks), and General Awareness (25 questions, 50 marks). "
            "Negative marking of 0.50 marks applies for each wrong answer in Tier-1. "
            "Tier-2 is also Computer Based and has three modules. "
            "Module-I covers Mathematical Abilities and Reasoning with 60 questions in 60 minutes. "
            "Module-II covers English Language and General Awareness with 85 questions in 60 minutes. "
            "Module-III is the Skill Test or Typing Test which is qualifying in nature. "
            "Tier-2 has negative marking of 1 mark per wrong answer in Modules I and II. "
            "Candidates should always check the official SSC notification for exact vacancy and cut-off details."
        ),
    },
    {
        "id": "doc_002",
        "topic": "SSC CHSL Typing Test",
        "text": (
            "The SSC CHSL Typing Test is part of Tier-2 Module-III and is qualifying in nature. "
            "For LDC and JSA posts, candidates need 35 words per minute in English or 30 words per minute in Hindi. "
            "For Data Entry Operator posts, a speed of 8000 key depressions per hour is required. "
            "For DEO in the CAG office, the required speed is 15000 key depressions per hour. "
            "The test is conducted on a computer and errors must stay within the allowed limit. "
            "Recommended practice platforms include TypingMaster, keybr.com, and 10fastfingers.com. "
            "Daily practice of at least 30 minutes is recommended to reach the required speed. "
            "Hindi typing candidates must use Krutidev or Unicode font as prescribed by SSC. "
            "The typing test does not carry marks but must be cleared for final selection. "
            "Candidates should practice typing passages similar to the ones used in the exam."
        ),
    },
    {
        "id": "doc_003",
        "topic": "SSC CHSL Detailed Syllabus",
        "text": (
            "The SSC CHSL Tier-1 syllabus covers four sections in detail. "
            "English Language includes Reading Comprehension, Cloze Test, Para Jumbles, Fill in the Blanks, "
            "Sentence Improvement, Error Detection, Synonyms, Antonyms, One Word Substitution, Idioms and Phrases. "
            "General Intelligence and Reasoning covers Analogy, Classification, Series, Coding-Decoding, "
            "Blood Relations, Direction and Distance, Venn Diagrams, Syllogisms, Matrix, and Non-Verbal Reasoning. "
            "Quantitative Aptitude covers Number System, Percentages, Ratio and Proportion, Average, "
            "Simple and Compound Interest, Profit and Loss, Discount, Time and Work, Time and Distance, "
            "Mensuration in 2D and 3D, Basic Trigonometry, and Algebra. "
            "General Awareness includes History, Geography, Polity, Economics, Science topics like Physics, "
            "Chemistry and Biology, Current Events, Sports, Books and Authors, Important Days, and Awards. "
            "Tier-2 Module-I covers advanced Mathematics and Reasoning. "
            "Tier-2 Module-II covers advanced English and General Awareness including Banking and Economics basics. "
            "Solving previous year papers is highly recommended for understanding question patterns and difficulty."
        ),
    },
    {
        "id": "doc_004",
        "topic": "Banking Exam Overview",
        "text": (
            "Banking exams in India are conducted by IBPS and individual banks like SBI and RBI. "
            "IBPS PO and IBPS Clerk are the most popular banking exams for graduates including B.Tech students. "
            "IBPS PO has three stages: Preliminary Exam, Mains Exam, and Interview. "
            "The Prelims has English Language, Quantitative Aptitude, and Reasoning Ability with 100 questions in 60 minutes. "
            "The Mains has Reasoning and Computer Aptitude, General and Banking Awareness, English Language, "
            "and Data Analysis and Interpretation for 200 marks. "
            "SBI PO follows a similar pattern with Prelims, Mains, and Group Exercise plus Interview. "
            "IBPS Clerk is for clerical positions and has Prelims and Mains but no interview stage. "
            "RBI Grade B is a prestigious exam with Phase I objective, Phase II descriptive and objective, and Interview. "
            "Key topics across all banking exams include Quantitative Aptitude, Reasoning, English, "
            "General Awareness, Current Affairs, and Computer Knowledge. "
            "Candidates must follow current affairs for at least 6 months before the exam, "
            "especially banking news, RBI policies, and Indian economy updates."
        ),
    },
    {
        "id": "doc_005",
        "topic": "DSA Preparation Roadmap",
        "text": (
            "Data Structures and Algorithms is the most critical skill for software engineering placements. "
            "Start with Arrays and Strings in the first two weeks covering two-pointer, sliding window, and prefix sum. "
            "Week three covers Linked Lists including singly linked, doubly linked, cycle detection, and reversal. "
            "Week four covers Stacks and Queues including implementation and monotonic stack problems. "
            "Weeks five and six cover Trees and Binary Trees including traversals, BST, LCA, and diameter problems. "
            "Weeks seven and eight cover Graphs including BFS, DFS, Dijkstra, topological sort, and union-find. "
            "Weeks nine through eleven cover Dynamic Programming including knapsack, LCS, LIS, and matrix chain. "
            "Week twelve covers Sorting and Searching including merge sort, quick sort, and binary search variations. "
            "The primary practice platform is LeetCode. GeeksforGeeks and Codeforces are also useful. "
            "Target solving 150 to 200 problems on LeetCode before attending interviews. "
            "Focus on Easy problems first, then Medium problems, and only a few Hard problems. "
            "Company-wise problem sets are available on LeetCode for top product and service companies."
        ),
    },
    {
        "id": "doc_006",
        "topic": "Aptitude Preparation Strategy",
        "text": (
            "Quantitative Aptitude is tested in almost all placement drives and competitive exams. "
            "Key topics include Number System, HCF and LCM, Percentages, Profit and Loss, "
            "Simple and Compound Interest, Time and Work, Time Speed Distance, Ratio and Proportion, "
            "Mixtures and Alligation, Mensuration, and Data Interpretation. "
            "For placements also cover Probability, Permutation and Combination, Progressions, and Logarithms. "
            "Recommended books are R.S. Aggarwal Quantitative Aptitude and Arun Sharma Quantitative Aptitude. "
            "For Logical Reasoning cover syllogisms, blood relations, coding-decoding, seating arrangements, and puzzles. "
            "Daily practice of 20 to 30 questions is recommended. "
            "Take at least one full mock test every weekend and analyze your performance thoroughly. "
            "For campus placements, IndiaBix, PrepInsta, and TCS iON Campus Commune are good platforms. "
            "Focus on both speed and accuracy since tests are time-bound and may have negative marking. "
            "Identify weak areas through test analysis and revisit those topics with targeted practice."
        ),
    },
    {
        "id": "doc_007",
        "topic": "Resume Tips for B.Tech Students",
        "text": (
            "A strong resume is the first step to landing placements and internships for B.Tech students. "
            "Keep your resume to one page for freshers and use a clean ATS-friendly format. "
            "The section order should be Contact Info, Objective or Summary, Education, Skills, "
            "Projects, Internships, Achievements, and Certifications. "
            "In the Skills section list programming languages like C++, Python, and Java, "
            "frameworks like React, Django, and Spring, and tools like Git, Docker, and SQL. "
            "Projects are the most important section for freshers. Describe each project in two to three bullet points "
            "using the STAR format covering Situation, Task, Action, and Result. "
            "Include GitHub links and live demo links wherever possible. "
            "Quantify achievements wherever possible such as reduced load time by 40 percent or handled 10000 API requests per day. "
            "Avoid spelling mistakes, passive voice, and generic phrases like hardworking team player. "
            "Use action verbs such as Developed, Designed, Implemented, Optimized, Deployed, and Automated. "
            "Get your resume reviewed by seniors or use tools like Resumeworded or Jobscan. "
            "Tailor your resume for each company by matching keywords from the job description."
        ),
    },
    {
        "id": "doc_008",
        "topic": "Interview Preparation",
        "text": (
            "Campus placement interviews typically have a Technical Round, HR Round, and sometimes a Managerial Round. "
            "In the Technical Round expect questions on your preferred programming language, "
            "OOPs concepts like inheritance, polymorphism, encapsulation, and abstraction, "
            "DBMS topics like normalization, SQL queries, and transactions, "
            "OS topics like process scheduling, memory management, and deadlocks, "
            "and CN topics like OSI model, TCP/IP, and HTTP. "
            "DSA coding problems are asked in most product companies so practice on LeetCode and HackerRank. "
            "Be ready to explain all projects on your resume including design decisions and challenges faced. "
            "In the HR Round common questions include Tell me about yourself, Why this company, "
            "Where do you see yourself in 5 years, What are your strengths and weaknesses, "
            "and Describe a challenge you overcame. "
            "Prepare structured answers using the STAR method. Research the company before the interview. "
            "Practice mock interviews with friends or use platforms like Pramp and Interviewing.io. "
            "Dress formally, maintain eye contact, speak clearly, and ask thoughtful questions at the end."
        ),
    },
    {
        "id": "doc_009",
        "topic": "Government Job vs Private IT Job",
        "text": (
            "Choosing between a government job and a private IT job is a common dilemma for B.Tech graduates. "
            "Government jobs through SSC, Banking, and PSU exams offer job security and permanence, "
            "fixed working hours typically from 9 AM to 5 PM, pension and retirement benefits, "
            "medical benefits for the entire family, housing rent allowance, paid leave, "
            "gradual but guaranteed salary increments through pay commissions, and social prestige. "
            "Private IT jobs offer higher starting salaries typically ranging from 3.5 to 12 LPA for freshers, "
            "fast-paced career growth, opportunity to work on cutting-edge technology, performance-based bonuses, "
            "remote work flexibility in many companies, global exposure and foreign travel opportunities, "
            "and stock options in startups. "
            "Government jobs require clearing competitive exams which can take 1 to 3 years of dedicated preparation. "
            "Private IT jobs require strong DSA, programming, and aptitude skills developed during B.Tech. "
            "The right choice depends on your priorities of financial security versus growth potential "
            "and stability versus dynamism. Many students prepare for both simultaneously in their final year."
        ),
    },
    {
        "id": "doc_010",
        "topic": "Study Planning Strategy",
        "text": (
            "An effective study plan is crucial for success in competitive exams and campus placements. "
            "Step one is to analyze the syllabus completely and list all topics. "
            "Step two is to assess your current level in each topic honestly. "
            "Step three is to allocate time based on topic weightage and your personal weak areas. "
            "A typical daily schedule for a B.Tech student could be as follows. "
            "From 6 AM to 8 AM practice DSA on LeetCode. "
            "From 9 AM to 5 PM attend college and classes. "
            "From 6 PM to 8 PM revise aptitude or subject topics. "
            "From 8 PM to 9 PM study current affairs or research companies. "
            "Take at least one full mock test every Saturday or Sunday and analyze it thoroughly. "
            "Revise all topics covered in a month at the end of each month. "
            "Use the Pomodoro Technique of 25 minutes focused study followed by a 5 minute break. "
            "Maintain a notes document for quick revision before exams or interviews. "
            "Set SMART goals that are Specific, Measurable, Achievable, Relevant, and Time-bound. "
            "Track daily and weekly progress in a journal or spreadsheet."
        ),
    },
    {
        "id": "doc_011",
        "topic": "Time Management for Students",
        "text": (
            "Time management is the most important differentiating factor for placement success. "
            "Avoid spending all time on college assignments while neglecting placement preparation. "
            "Use time-blocking by assigning specific time slots for DSA, aptitude, projects, and academics weekly. "
            "Prioritize using the Eisenhower Matrix by doing Urgent plus Important tasks first, "
            "planning Important but Not Urgent tasks, and delegating or dropping the rest. "
            "Limit social media and entertainment to defined break times using apps like Forest or StayFocusd. "
            "Batch similar tasks together such as answering all messages at once or studying related topics together. "
            "Reserve the two most productive hours of your day, typically morning, for your hardest subject. "
            "Avoid multitasking as it reduces efficiency significantly. "
            "Review your weekly schedule every Sunday evening and adjust for the upcoming week. "
            "Use commute time for listening to revision audio, podcasts, or current affairs. "
            "Create accountability by sharing goals with a study partner or joining a study group."
        ),
    },
    {
        "id": "doc_012",
        "topic": "Common Placement Mistakes",
        "text": (
            "Many B.Tech students make avoidable mistakes that cost them placement opportunities. "
            "Mistake one is starting DSA preparation too late. Ideally start from third year itself. "
            "Mistake two is memorizing DSA solutions without understanding the underlying logic and patterns. "
            "Mistake three is not practicing under timed conditions since real tests have strict time limits. "
            "Mistake four is skipping aptitude preparation assuming only coding matters. "
            "Aptitude is the first filter in most mass-hiring placement drives. "
            "Mistake five is having a weak resume with vague project descriptions and no quantifiable achievements. "
            "Mistake six is not researching the company before the interview. "
            "Knowing the company product and recent news is essential for HR and managerial rounds. "
            "Mistake seven is getting nervous and freezing in interviews. Practice thinking out loud. "
            "Mistake eight is applying only to top companies and ignoring mid-tier companies with great learning. "
            "Mistake nine is neglecting soft skills since communication and attitude matter greatly in HR rounds. "
            "Mistake ten is comparing yourself to peers and losing motivation. Every student has a different journey. "
            "Consistent daily effort over 6 to 12 months is far more effective than last-minute cramming."
        ),
    },
    {
        "id": "doc_013",
        "topic": "Internship Preparation",
        "text": (
            "Internships are crucial for B.Tech students as they provide real-world experience and often lead to PPOs. "
            "Start applying for internships from second year itself to build experience early. "
            "Types of internships available include Software Development, Data Science and ML, "
            "Web Development, Embedded Systems, and Research Internships. "
            "To get internships at top companies like Google, Microsoft, Amazon, and Flipkart you need "
            "a strong competitive programming profile on Codeforces or LeetCode, "
            "relevant projects on GitHub, a well-written resume, and active networking on LinkedIn. "
            "For off-campus internships apply via LinkedIn, Internshala, AngelList, and company career pages. "
            "The typical selection process includes an Online Coding Test, Technical Interview, and HR Interview. "
            "During the internship communicate proactively with your mentor, ask questions, and document your work. "
            "Aim to make a measurable contribution to the project you are assigned. "
            "Convert your internship to a Pre-Placement Offer by exceeding expectations and building good relationships. "
            "Even small internships at startups are valuable since the experience and learning matter. "
            "Document your internship learnings and add the experience with quantified impact on your resume."
        ),
    },
    {
        "id": "doc_014",
        "topic": "Placement Timeline and Schedule",
        "text": (
            "A well-structured placement preparation timeline for B.Tech students works as follows. "
            "In third year Semester 5 from July to November, begin DSA basics covering arrays, strings, "
            "linked lists, stacks, queues, and trees. Solve 50 or more LeetCode easy problems. "
            "Build one full-stack or domain-specific project during this period. "
            "In third year Semester 6 from January to May, continue DSA with graphs, DP, and advanced trees. "
            "Solve 50 or more medium problems. Prepare your resume and get it reviewed. Apply for summer internships. "
            "In fourth year Semester 7 from July to September this is the peak placement season. "
            "Complete 150 or more LeetCode problems. Practice aptitude daily. Attend mock placement drives. "
            "Prepare core CS subjects including DBMS, OS, Computer Networks, and OOPs. "
            "Attend company Pre-Placement Talks to understand each company's process. "
            "In fourth year Semester 8 from October to December attend off-campus drives, use referrals, "
            "and apply on job portals if on-campus placement is not yet complete. "
            "For SSC and Banking aspirants start exam preparation in third year alongside college work. "
            "Always maintain a Plan A for on-campus placement, Plan B for off-campus, "
            "and Plan C for higher studies or government exams."
        ),
    },
]

# ── Singletons ────────────────────────────────────────────────────────────────

_collection = None
_model = None


def get_rag_components():
    global _collection, _model

    if _collection is not None:
        return _collection, _model

    _model = SentenceTransformer("all-MiniLM-L6-v2")

    persist_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chroma_db")
    os.makedirs(persist_dir, exist_ok=True)

    client = chromadb.PersistentClient(path=persist_dir)

    existing_names = [c.name for c in client.list_collections()]

    if "career_kb" in existing_names:
        _collection = client.get_collection("career_kb")
    else:
        _collection = client.create_collection(
            name="career_kb",
            metadata={"hnsw:space": "cosine"},
        )
        texts = [d["text"] for d in DOCUMENTS]
        ids = [d["id"] for d in DOCUMENTS]
        metadatas = [{"topic": d["topic"]} for d in DOCUMENTS]
        embeddings = _model.encode(texts, show_progress_bar=False).tolist()
        _collection.add(
            documents=texts,
            ids=ids,
            metadatas=metadatas,
            embeddings=embeddings,
        )

    return _collection, _model


def retrieve(query: str, n_results: int = 3) -> Tuple[str, List[str]]:
    collection, model = get_rag_components()
    query_embedding = model.encode([query], show_progress_bar=False).tolist()
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=n_results,
        include=["documents", "metadatas", "distances"],
    )
    docs = results["documents"][0]
    metas = results["metadatas"][0]
    distances = results["distances"][0]

    chunks = []
    sources = []
    for doc, meta, dist in zip(docs, metas, distances):
        topic = meta.get("topic", "General")
        relevance = round(1.0 - dist, 3)
        chunks.append(f"[{topic}] (relevance: {relevance})\n{doc}")
        sources.append(topic)

    return "\n\n---\n\n".join(chunks), sources
