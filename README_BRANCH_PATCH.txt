This branch patch does four things:
1. Uses snake_case uniformly across the backend.
2. Replaces the Random Forest app model with a simple Sequential dense network.
3. Adds a multilingual UI system with one shared layout and Spanish fallback.
4. Adds crypto_demo.py to demonstrate hashing, RSA signing, verification, and tamper detection.

Run order:
1. python generate_data.py
2. python train_sequential.py
3. python generate_monitoring_data.py
4. python app.py

Or simply:
python run_demo.py
