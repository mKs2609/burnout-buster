"""generate_dataset.py — run once to create burnout_dataset.csv"""
import numpy as np, pandas as pd
np.random.seed(42)

def make(n, risk):
    if risk=="High":
        e=np.clip(np.random.normal(6,1,n),4,8).astype(int); a=np.clip(np.random.normal(8,1.5,n),5,12).astype(int)
        sl=np.clip(np.random.normal(4.5,1,n),3,6).astype(int); cg=np.round(np.clip(np.random.normal(5.5,.8,n),4,7),1)
        bl=np.clip(np.random.poisson(3,n),1,8).astype(int); fo=np.clip(np.random.normal(8,1,n),6,10).astype(int)
        pp=np.clip(np.random.normal(8,1,n),6,10).astype(int); fe=np.clip(np.random.normal(8,1,n),6,10).astype(int)
        sm=np.clip(np.random.normal(7,1.5,n),4,12).astype(int); rs=np.clip(np.random.normal(8,1,n),5,10).astype(int)
        ex=np.clip(np.random.normal(1,1,n),0,3).astype(int); dq=np.clip(np.random.normal(3,1.5,n),1,6).astype(int)
        co=np.clip(np.random.normal(3,1.5,n),1,5).astype(int); su=np.clip(np.random.normal(3,1.5,n),1,5).astype(int)
        at=np.clip(np.random.normal(8,1,n),5,10).astype(int); st=np.clip(np.random.normal(3,1.5,n),1,6).astype(int)
        mh=np.clip(np.random.poisson(.5,n),0,3).astype(int)
    elif risk=="Low":
        e=np.clip(np.random.normal(2,1,n),1,4).astype(int); a=np.clip(np.random.normal(3,1,n),1,6).astype(int)
        sl=np.clip(np.random.normal(8,1,n),6,10).astype(int); cg=np.round(np.clip(np.random.normal(8.5,.8,n),7,10),1)
        bl=np.clip(np.random.poisson(.2,n),0,2).astype(int); fo=np.clip(np.random.normal(3,1.5,n),1,5).astype(int)
        pp=np.clip(np.random.normal(3,1.5,n),1,5).astype(int); fe=np.clip(np.random.normal(4,1.5,n),1,6).astype(int)
        sm=np.clip(np.random.normal(2,1,n),0,4).astype(int); rs=np.clip(np.random.normal(3,1.5,n),1,5).astype(int)
        ex=np.clip(np.random.normal(5,1,n),3,7).astype(int); dq=np.clip(np.random.normal(7,1.5,n),5,10).astype(int)
        co=np.clip(np.random.normal(8,1,n),6,10).astype(int); su=np.clip(np.random.normal(8,1,n),6,10).astype(int)
        at=np.clip(np.random.normal(4,1.5,n),1,6).astype(int); st=np.clip(np.random.normal(7,1.5,n),5,12).astype(int)
        mh=np.clip(np.random.poisson(.1,n),0,1).astype(int)
    else:
        e=np.clip(np.random.normal(4,1.5,n),2,7).astype(int); a=np.clip(np.random.normal(5,1.5,n),2,9).astype(int)
        sl=np.clip(np.random.normal(6,1,n),4,8).astype(int); cg=np.round(np.clip(np.random.normal(7,.8,n),5,9),1)
        bl=np.clip(np.random.poisson(1.2,n),0,4).astype(int); fo=np.clip(np.random.normal(5.5,1.5,n),3,8).astype(int)
        pp=np.clip(np.random.normal(5.5,1.5,n),3,8).astype(int); fe=np.clip(np.random.normal(6,1.5,n),3,9).astype(int)
        sm=np.clip(np.random.normal(4,1.5,n),1,8).astype(int); rs=np.clip(np.random.normal(5,1.5,n),2,8).astype(int)
        ex=np.clip(np.random.normal(3,1.5,n),0,6).astype(int); dq=np.clip(np.random.normal(5,1.5,n),2,8).astype(int)
        co=np.clip(np.random.normal(5,1.5,n),2,8).astype(int); su=np.clip(np.random.normal(5,1.5,n),2,8).astype(int)
        at=np.clip(np.random.normal(6,1.5,n),3,9).astype(int); st=np.clip(np.random.normal(5,1.5,n),2,9).astype(int)
        mh=np.clip(np.random.poisson(.3,n),0,2).astype(int)
    return pd.DataFrame({"exams_per_month":e,"assignments_per_week":a,"attendance_pressure":at,
        "cgpa":cg,"backlogs":bl,"study_hours_per_day":st,"fomo_score":fo,"peer_pressure":pp,
        "family_expectations":fe,"social_media_hrs":sm,"rejection_sensitivity":rs,"sleep_hours":sl,
        "exercise_days":ex,"diet_quality":dq,"confidence":co,"support_system":su,
        "mental_health_visits":mh,"burnout_risk":risk})

df=pd.concat([make(100,"High"),make(100,"Medium"),make(100,"Low")]).sample(frac=1,random_state=42).reset_index(drop=True)
df.to_csv("burnout_dataset.csv",index=False)
print("Dataset saved:", df["burnout_risk"].value_counts().to_dict())
