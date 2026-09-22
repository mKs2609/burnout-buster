"""advice.py — the action plan shown with each result."""

TIPS = {
    "sleep_hours":           ("Sleep First", "Sleep is one of the biggest factors in your score. Even one extra hour helps — try a consistent bedtime for just 2 weeks."),
    "social_media_hrs":      ("Digital Reset", "Hours of scrolling quietly drain your energy. Try a daily limit with your phone's screen-time settings — you'll feel it within days."),
    "fomo_score":            ("Let Go of FOMO", "Constantly feeling like you're missing out is exhausting. You're on your own timeline, and that's okay."),
    "exercise_days":         ("Move Your Body", "Even a 20-minute walk 3 times a week noticeably reduces academic stress."),
    "confidence":            ("Rebuild Confidence", "Low self-belief amplifies every other stressor. Write down 3 small wins each evening."),
    "support_system":        ("Reach Out", "You seem to be carrying a lot alone. One honest conversation with someone you trust — or your counselor — can shift a lot."),
    "backlogs":              ("Clear the Backlog", "Pending subjects create constant background anxiety. Ask your academic advisor for a realistic plan this week."),
    "rejection_sensitivity": ("Be Kinder to Yourself", "Setbacks feel personal when you're stretched thin. Treat a rejection as one data point, not a verdict on you."),
    "assignments_per_week":  ("Tame the Workload", "Heavy weekly load? List everything due, pick the top 3 each day, and ask early if a deadline is unrealistic."),
    "exams_per_month":       ("Plan Exam Weeks", "Frequent exams leave no recovery time. Block short daily revision slots so exam weeks aren't all-nighters."),
    "attendance_pressure":   ("Talk About Attendance", "If attendance worries you, check exactly where you stand and speak to your mentor before it snowballs."),
    "cgpa":                  ("Grades Aren't Everything", "Your CGPA is one number. Focus on one subject to improve this term, and ask faculty for help early."),
    "peer_pressure":         ("Set Your Own Pace", "Comparing yourself with others raises stress. Pick goals that matter to you, not to the group chat."),
    "family_expectations":   ("Share the Load", "High expectations at home are hard to carry silently. Consider telling your family how you're really doing."),
    "diet_quality":          ("Fuel Up", "Skipped or rushed meals affect mood and focus. Start with a proper breakfast on class days."),
    "study_hours_per_day":   ("Balance Study Time", "Very long or very short study days both raise stress. Aim for focused blocks with real breaks."),
    "mental_health_visits":  ("Keep Talking", "Reaching out is a strength. Keep in touch with your counselor about what's helping."),
}

def personalized_advice(feat_dict, label, drivers=()):
    # Tips follow the model's explanation first, then simple rules
    tips = [TIPS[f] for f,_ in drivers if f in TIPS]
    rules = [("sleep_hours", float(feat_dict.get("sleep_hours",7))<6),
             ("social_media_hrs", float(feat_dict.get("social_media_hrs",3))>5),
             ("support_system", float(feat_dict.get("support_system",5))<4),
             ("backlogs", float(feat_dict.get("backlogs",0))>2)]
    for f,hit in rules:
        if hit and TIPS[f] not in tips: tips.append(TIPS[f])
    if not tips:
        if label=="Thriving":
            tips.append(("Keep It Going", "All your indicators look healthy. Keep your current habits — you're doing better than you think."))
        else:
            tips.append(("Small Steps", "Multiple areas need attention. Start with just one thing this week — better sleep is always the highest-impact first step."))
    return tips[:3]
