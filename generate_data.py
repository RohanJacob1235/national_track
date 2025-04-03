import pandas as pd
import random
import faker

fake = faker.Faker()

# Define parameters
num_participants = 400
days = [1, 2, 3, 4]
tracks = ["Track 1", "Track 2", "Track 3", "Track 4"]
states = ["Kerala", "Tamil Nadu", "Karnataka", "Maharashtra"]
colleges = ["College A", "College B", "College C", "College D"]
engagement_levels = ["Low", "Medium", "High"]

# Generate random dataset
data = []
for i in range(1, num_participants + 1):
    data.append({
        "Participant_ID": i,
        "Name": fake.name(),
        "College": random.choice(colleges),
        "State": random.choice(states),
        "Day": random.choice(days),
        "Track": random.choice(tracks),
        "Feedback": fake.sentence(nb_words=10),
        "Presentation_Score": round(random.uniform(5, 10), 1),
        "Engagement_Level": random.choice(engagement_levels),
        "Photo_Filename": f"day{random.choice(days)}_photo_{i}.jpg"
    })

df = pd.DataFrame(data)

# Save dataset
df.to_csv("data/poster_presentation_data.csv", index=False)
print("✅ Dataset generated and saved as 'data/poster_presentation_data.csv'")
