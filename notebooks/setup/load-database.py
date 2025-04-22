import uuid
import psycopg2
from faker import Faker
from sentence_transformers import SentenceTransformer
import random
from datetime import datetime

# --- Config ---
NUM_STUDENTS = 100
MODEL_NAME = "notebooks/model/all-MiniLM-L6-v2"

# --- Init ---
fake = Faker()
model = SentenceTransformer(MODEL_NAME)

# --- DB Connection ---
conn = psycopg2.connect(
    dbname="auth_db",
    user="user",
    password="password",
    host="localhost",
    port=5432)
cur = conn.cursor()

# --- Sample Data Pools ---
schools = [
    "Harvard", "Stanford", "MIT", "UCLA", "Michigan", "NYU",
    "Princeton", "Yale", "Columbia", "UC Berkeley", "University of Chicago",
    "Duke", "University of Pennsylvania", "Caltech", "Northwestern", "Cornell"
]
majors = ["Computer Science", "Economics", "Biology",
          "Design", "Political Science", "Physics",
          "Engineering", "History", "Chemistry",
          "Environmental Science", "Music", "Sociology",
          "Anthropology", "Statistics", "Marketing",
          "Journalism"]
minors = ["Math", "Philosophy", "Business",
          "Data Science", "Psychology", "Art History",
          "Astronomy", "Linguistics", "Theater", "Education",
          "Public Health", "Geography", "International Relations",
          "Graphic Design", "Creative Writing"]
degrees = ["Bachelors", "Masters"]
sports_list = [
    "Rowing", "Basketball", "Soccer", "Tennis", "Swimming",
    "Baseball", "Volleyball", "Track and Field", "Golf", "Hockey",
    "Lacrosse", "Wrestling", "Cycling", "Skiing", "Badminton"
]
activities_list = [
    "Student Government", "Robotics Club", "Drama Club", "Debate Team", "Hackathons",
    "Chess Club", "Photography Club", "Volunteer Work", "Music Band", "Art Club",
    "Science Club", "Math Club", "Environmental Club", "Dance Team", "Book Club"
]
keywords_pool = ["teamwork", "leadership", "AI",
                 "sustainability", "startups", "research", "design",
                 "innovation", "entrepreneurship", "machine learning",
                 "data analysis", "cloud computing", "cybersecurity",
                 "blockchain", "public speaking", "networking", "problem-solving"]
image_urls = [f"https://randomuser.me/api/portraits/men/{i}.jpg" for i in range(20)] + \
             [f"https://randomuser.me/api/portraits/women/{i}.jpg" for i in range(20)]


def make_student():
    name = fake.name()
    school = random.choice(schools)
    major = random.choice(majors)
    minor = random.choice(minors)
    school_location = f"{fake.city()}, {fake.state_abbr()}"
    degree_type = random.choice(degrees)
    graduation_year = random.randint(2024, 2027)
    home_town = fake.city()
    home_state = fake.state()
    sports = random.sample(sports_list, k=random.randint(0, 2))
    activities = random.sample(activities_list, k=random.randint(1, 3))
    keywords = random.sample(keywords_pool, k=random.randint(2, 4))
    bio = fake.sentence(nb_words=12)
    image = random.choice(image_urls)

    # Embed profile
    profile_text = (
        f"{name} is studying {major} with a minor in {minor} at {school} in {school_location}. "
        f"They are pursuing a {degree_type} degree and will graduate in {graduation_year}. "
        f"Activities: {', '.join(activities)}. Sports: {', '.join(sports)}. "
        f"From {home_town}, {home_state}. Interests: {', '.join(keywords)}. {bio}"
    )
    embedding = model.encode(profile_text).tolist()

    return (
        str(uuid.uuid4()), name, keywords, school, major, minor, school_location,
        degree_type, sports, activities, graduation_year, home_town, home_state,
        bio, embedding, image
    )


# --- Insert Students ---
for _ in range(NUM_STUDENTS):
    student = make_student()
    print(
        f"Creating student: {student[1]} - {student[3]} - {student[4]} - {student[5]}")
    cur.execute("""
        INSERT INTO students (
            id, name, keywords, school, major, minor, school_location,
            degree_type, sports, campus_activities, graduation_year,
            home_town, home_state, bio, embedding, image
        ) VALUES (
            %s, %s, %s, %s, %s, %s, %s,
            %s, %s, %s, %s,
            %s, %s, %s, %s, %s
        )
    """, student)


print(f"✅ Inserted {NUM_STUDENTS} fake students into the database.")
conn.commit()
cur.close()
conn.close()
