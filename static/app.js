/**
 * FIFA World Cup 2026 Match Predictor — Frontend Logic
 * =====================================================
 * Handles team loading, prediction API calls, and result rendering.
 */

// Cache for teams data
let teamsData = [];

// -----------------------------------------------------------------------
// 1. Initialize — load teams on page load
// -----------------------------------------------------------------------

document.addEventListener("DOMContentLoaded", () => {
    loadTeams();
});

async function loadTeams() {
    try {
        const res = await fetch("/api/teams");
        const data = await res.json();
        teamsData = data.teams;
        populateDropdowns(teamsData);
    } catch (err) {
        showError("Failed to load teams. Is the server running?");
        console.error(err);
    }
}

function populateDropdowns(teams) {
    const team1 = document.getElementById("team1-select");
    const team2 = document.getElementById("team2-select");

    // Clear existing options except the placeholder
    team1.innerHTML = '<option value="">Select Team...</option>';
    team2.innerHTML = '<option value="">Select Team...</option>';

    teams.forEach(t => {
        const opt1 = document.createElement("option");
        opt1.value = t.name;
        opt1.textContent = `${t.name} (Rank #${t.rank})`;
        team1.appendChild(opt1);

        const opt2 = opt1.cloneNode(true);
        team2.appendChild(opt2);
    });

    // Add change listeners for rank display
    team1.addEventListener("change", () => updateRankDisplay("team1"));
    team2.addEventListener("change", () => updateRankDisplay("team2"));
}

function updateRankDisplay(which) {
    const select = document.getElementById(`${which}-select`);
    const rankDiv = document.getElementById(`${which}-rank`);
    const teamName = select.value;

    if (!teamName) {
        rankDiv.textContent = "";
        return;
    }

    const team = teamsData.find(t => t.name === teamName);
    if (team) {
        rankDiv.textContent = `FIFA Rank: #${team.rank} • ${team.confederation} • ${team.points.toFixed(0)} pts`;
    }
}

// -----------------------------------------------------------------------
// 2. Predict
// -----------------------------------------------------------------------

async function predict() {
    const team1 = document.getElementById("team1-select").value;
    const team2 = document.getElementById("team2-select").value;
    const stage = document.getElementById("stage-select").value;

    // Validate
    if (!team1 || !team2) {
        showError("Please select both teams.");
        return;
    }
    if (team1 === team2) {
        showError("Please select two different teams.");
        return;
    }

    // Show loading
    hideError();
    hideResults();
    showLoading();

    try {
        const res = await fetch("/api/predict", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ team1, team2, stage }),
        });

        const data = await res.json();
        hideLoading();

        if (data.error) {
            showError(data.error);
            return;
        }

        renderResults(data);
    } catch (err) {
        hideLoading();
        showError("Prediction failed. Check server connection.");
        console.error(err);
    }
}

// -----------------------------------------------------------------------
// 3. Render Results
// -----------------------------------------------------------------------

function renderResults(data) {
    const el = document.getElementById("results");

    // Winner banner
    const bannerColor = data.winner === "Draw" ? "var(--warning)" : "var(--success)";
    const textColor = data.winner === "Draw" ? "var(--border)" : "var(--border)";
    const banner = document.getElementById("winner-banner");
    banner.style.background = bannerColor;
    banner.style.color = textColor;
    document.getElementById("winner-name").textContent = data.winner;
    document.getElementById("winner-confidence").textContent =
        `${data.confidence}% confidence`;

    // Score
    document.getElementById("score-team1-name").textContent = data.team1;
    document.getElementById("score-team2-name").textContent = data.team2;
    document.getElementById("score-team1").textContent = data.score[data.team1];
    document.getElementById("score-team2").textContent = data.score[data.team2];

    // Probabilities
    const p1 = data.probabilities[data.team1];
    const pDraw = data.probabilities["Draw"];
    const p2 = data.probabilities[data.team2];

    document.getElementById("prob-team1-label").textContent = data.team1;
    document.getElementById("prob-team2-label").textContent = data.team2;
    document.getElementById("prob-team1-val").textContent = `${p1}%`;
    document.getElementById("prob-draw-val").textContent = `${pDraw}%`;
    document.getElementById("prob-team2-val").textContent = `${p2}%`;

    // Animate bars
    setTimeout(() => {
        document.getElementById("prob-bar-team1").style.width = `${p1}%`;
        document.getElementById("prob-bar-draw").style.width = `${pDraw}%`;
        document.getElementById("prob-bar-team2").style.width = `${p2}%`;
    }, 50);

    // MOTM
    const motm = data.motm;
    document.getElementById("motm-name").textContent = motm.name;
    document.getElementById("motm-details").innerHTML =
        `<span>${motm.position}</span> <span>${motm.club}</span> <span>${motm.team}</span>`;
    document.getElementById("motm-reason").textContent = `"${motm.reason}"`;
    document.getElementById("motm-rating").textContent = `RATING: ${motm.rating}/10`;

    // Stats
    document.getElementById("stats-team1-name").textContent = data.team1;
    document.getElementById("stats-team2-name").textContent = data.team2;

    const statsGrid = document.getElementById("stats-grid");
    statsGrid.innerHTML = "";

    const statsList = [
        { key: "possession", label: "Possession (%)", suffix: "%" },
        { key: "shots", label: "Shots", suffix: "" },
        { key: "shots_on_target", label: "Shots on Target", suffix: "" },
        { key: "xg", label: "Expected Goals (xG)", suffix: "" },
        { key: "pass_accuracy", label: "Pass Accuracy", suffix: "%" },
        { key: "corners", label: "Corners", suffix: "" },
        { key: "fouls", label: "Fouls", suffix: "" },
        { key: "yellow_cards", label: "Yellow Cards", suffix: "" },
        { key: "red_cards", label: "Red Cards", suffix: "" },
        { key: "offsides", label: "Offsides", suffix: "" },
    ];

    statsList.forEach(stat => {
        const val1 = data.stats.team1[stat.key];
        const val2 = data.stats.team2[stat.key];
        const row = document.createElement("div");
        row.className = "stat-row";

        const isHigherBetter = !["fouls", "yellow_cards", "red_cards", "offsides"].includes(stat.key);
        const v1Class = isHigherBetter && val1 > val2 ? "stat-highlight" :
                        !isHigherBetter && val1 < val2 ? "stat-highlight" : "";
        const v2Class = isHigherBetter && val2 > val1 ? "stat-highlight" :
                        !isHigherBetter && val2 < val1 ? "stat-highlight" : "";

        row.innerHTML = `
            <span class="stat-val"><span class="${v1Class}">${val1}${stat.suffix}</span></span>
            <span class="stat-name">${stat.label}</span>
            <span class="stat-val"><span class="${v2Class}">${val2}${stat.suffix}</span></span>
        `;
        statsGrid.appendChild(row);
    });

    el.style.display = "block";

    // Scroll to results
    el.scrollIntoView({ behavior: "smooth", block: "start" });
}

// -----------------------------------------------------------------------
// 4. UI Helpers
// -----------------------------------------------------------------------

function showLoading() {
    document.getElementById("loading").style.display = "block";
}

function hideLoading() {
    document.getElementById("loading").style.display = "none";
}

function showError(msg) {
    const el = document.getElementById("error-msg");
    el.textContent = msg;
    el.style.display = "block";
}

function hideError() {
    document.getElementById("error-msg").style.display = "none";
}

function hideResults() {
    document.getElementById("results").style.display = "none";
    // Reset probability bars
    document.getElementById("prob-bar-team1").style.width = "0%";
    document.getElementById("prob-bar-draw").style.width = "0%";
    document.getElementById("prob-bar-team2").style.width = "0%";
}
