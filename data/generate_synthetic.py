"""Generate synthetic IPL match dataset for model training/testing."""
import csv
import random
from datetime import datetime, timedelta

TEAMS = [
    "Mumbai Indians", "Chennai Super Kings", "Royal Challengers Bangalore",
    "Kolkata Knight Riders", "Delhi Capitals", "Rajasthan Royals",
    "Sunrisers Hyderabad", "Punjab Kings", "Gujarat Titans", "Lucknow Super Giants"
]

VENUES = [
    "Wankhede Stadium, Mumbai",
    "MA Chidambaram Stadium, Chennai",
    "M Chinnaswamy Stadium, Bangalore",
    "Eden Gardens, Kolkata",
    "Arun Jaitley Stadium, Delhi",
    "Sawai Mansingh Stadium, Jaipur",
    "Rajiv Gandhi Intl Stadium, Hyderabad",
    "Punjab Cricket Association Stadium, Mohali",
    "Narendra Modi Stadium, Ahmedabad",
    "Bharat Ratna Shri Atal Bihari Vajpayee Ekana Cricket Stadium, Lucknow",
]

TOSS_DECISIONS = ["bat", "field"]


def generate_matches(n: int = 50) -> list[dict]:
    matches = []
    start_date = datetime(2024, 3, 22)

    for i in range(1, n + 1):
        t1, t2 = random.sample(TEAMS, 2)
        venue = random.choice(VENUES)
        toss_winner = random.choice([t1, t2])
        toss_decision = random.choice(TOSS_DECISIONS)

        # Simulate a realistic-ish winner with slight home advantage
        home_team = TEAMS[VENUES.index(venue)] if venue in VENUES else None
        t1_prob = 0.5
        if t1 == home_team:
            t1_prob += 0.08
        elif t2 == home_team:
            t1_prob -= 0.08
        if toss_winner == t1:
            t1_prob += 0.04
        else:
            t1_prob -= 0.04

        winner = t1 if random.random() < t1_prob else t2
        win_margin = random.randint(1, 8) if random.random() < 0.5 else random.randint(5, 60)
        result_type = "wickets" if win_margin <= 10 else "runs"

        t1_score = random.randint(130, 220)
        t2_score = t1_score + random.randint(-40, 40)

        match_date = start_date + timedelta(days=(i - 1) * 2 + random.randint(0, 1))

        matches.append({
            "id": i,
            "season": 2024,
            "date": match_date.strftime("%Y-%m-%d"),
            "team1": t1,
            "team2": t2,
            "venue": venue,
            "toss_winner": toss_winner,
            "toss_decision": toss_decision,
            "winner": winner,
            "result_type": result_type,
            "win_margin": win_margin,
            "team1_score": t1_score,
            "team2_score": t2_score,
        })

    return matches


def main():
    random.seed(42)
    matches = generate_matches(50)

    fieldnames = list(matches[0].keys())
    with open("data/matches.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(matches)

    print(f"Generated {len(matches)} synthetic matches -> data/matches.csv")


if __name__ == "__main__":
    main()
