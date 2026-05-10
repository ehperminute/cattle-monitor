SQLite case-management upgrade

What this branch adds:
- SQLite persistence for cows, observations, notes, and follow-up actions
- Dashboard filters by status
- Cow detail page supports notes and actions
- Seed script imports the monitoring CSV into the SQLite database

Recommended workflow:
1. Keep your current app on main.
2. Create a branch for this experiment.
3. Copy these files into that branch.
4. Run:
   python generate_data.py
   python train.py
   python generate_monitoring_data.py
   python seed_db.py
   python app.py
