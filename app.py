from __future__ import annotations

from flask import Flask, abort, redirect, render_template, request, url_for

from db import execute, init_db, query_all, query_one

app = Flask(__name__)
init_db()


def latest_dashboard_rows(status_filter: str | None = None) -> list[dict]:
    sql = """
        SELECT o.*
        FROM observations o
        INNER JOIN (
            SELECT cow_id, MAX(observation_date) AS max_date
            FROM observations
            GROUP BY cow_id
        ) latest
        ON o.cow_id = latest.cow_id AND o.observation_date = latest.max_date
    """
    params: tuple = ()
    if status_filter:
        sql += " WHERE o.status = ?"
        params = (status_filter,)
    sql += " ORDER BY o.sick_probability DESC, o.cow_id ASC"

    rows = query_all(sql, params)
    return [dict(r) for r in rows]


def dashboard_summary(rows: list[dict]) -> dict:
    return {
        "total_cows": len(rows),
        "high_risk": sum(1 for r in rows if r["status"] == "High risk"),
        "review": sum(1 for r in rows if r["status"] == "Review"),
        "normal": sum(1 for r in rows if r["status"] == "Normal"),
    }


@app.route("/")
def index():
    status_filter = request.args.get("status", "").strip() or None
    rows = latest_dashboard_rows(status_filter)
    summary = dashboard_summary(rows)
    return render_template(
        "index.html",
        cows=rows,
        summary=summary,
        current_filter=status_filter or "All"
    )


@app.route("/cow/<cow_id>")
def cow_detail(cow_id: str):
    latest = query_one(
        """
        SELECT *
        FROM observations
        WHERE cow_id = ?
        ORDER BY observation_date DESC, id DESC
        LIMIT 1
        """,
        (cow_id,),
    )
    if latest is None:
        abort(404)

    history = query_all(
        """
        SELECT *
        FROM observations
        WHERE cow_id = ?
        ORDER BY observation_date DESC, id DESC
        """,
        (cow_id,),
    )

    notes = query_all(
        """
        SELECT *
        FROM case_notes
        WHERE cow_id = ?
        ORDER BY created_at DESC, id DESC
        """,
        (cow_id,),
    )

    actions = query_all(
        """
        SELECT *
        FROM follow_up_actions
        WHERE cow_id = ?
        ORDER BY created_at DESC, id DESC
        """,
        (cow_id,),
    )

    return render_template(
        "cow_detail.html",
        cow=dict(latest),
        history=[dict(r) for r in history],
        notes=[dict(r) for r in notes],
        actions=[dict(r) for r in actions],
    )


@app.post("/cow/<cow_id>/notes")
def add_note(cow_id: str):
    latest = query_one(
        """
        SELECT id
        FROM observations
        WHERE cow_id = ?
        ORDER BY observation_date DESC, id DESC
        LIMIT 1
        """,
        (cow_id,),
    )
    if latest is None:
        abort(404)

    note_type = (request.form.get("note_type") or "general").strip()
    note_text = (request.form.get("note_text") or "").strip()

    if note_text:
        execute(
            """
            INSERT INTO case_notes (cow_id, observation_id, note_type, note_text)
            VALUES (?, ?, ?, ?)
            """,
            (cow_id, latest["id"], note_type, note_text),
        )

    return redirect(url_for("cow_detail", cow_id=cow_id))


@app.post("/cow/<cow_id>/actions")
def add_action(cow_id: str):
    latest = query_one(
        """
        SELECT id
        FROM observations
        WHERE cow_id = ?
        ORDER BY observation_date DESC, id DESC
        LIMIT 1
        """,
        (cow_id,),
    )
    if latest is None:
        abort(404)

    action_type = (request.form.get("action_type") or "").strip()
    action_status = (request.form.get("status") or "open").strip()

    if action_type:
        execute(
            """
            INSERT INTO follow_up_actions (cow_id, observation_id, action_type, status)
            VALUES (?, ?, ?, ?)
            """,
            (cow_id, latest["id"], action_type, action_status),
        )

    return redirect(url_for("cow_detail", cow_id=cow_id))


@app.post("/actions/<int:action_id>/status")
def update_action_status(action_id: int):
    cow_id = request.form.get("cow_id", "")
    status = (request.form.get("status") or "open").strip()

    execute(
        "UPDATE follow_up_actions SET status = ? WHERE id = ?",
        (status, action_id),
    )

    return redirect(url_for("cow_detail", cow_id=cow_id))


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
