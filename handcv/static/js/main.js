function fetchStatus() {
    fetch('/status')
        .then(response => response.json())
        .then(data => {
            updateDashboard(data);
        })
        .catch(error => console.error('Error fetching status:', error));
}

function updateDashboard(data) {
    // Update Connection Status
    const statusDot = document.getElementById('status-dot');
    const statusText = document.getElementById('status-text');

    if (data.hand_detected) {
        statusDot.className = 'status-indicator active';
        statusText.textContent = 'Active Tracking';
        statusText.style.color = '#3b82f6';
    } else {
        statusDot.className = 'status-indicator inactive';
        statusText.textContent = 'Searching...';
        statusText.style.color = '#64748b';
    }

    // Update Big Stats
    document.getElementById('finger-count').textContent = data.finger_count;
    document.getElementById('hand-side').textContent = data.hand_label;

    // Update Finger Bars
    const container = document.getElementById('fingers-container');
    container.innerHTML = ''; // efficient enough for 5 items

    if (data.fingers && data.fingers.length > 0) {
        data.fingers.forEach(finger => {
            const fingerCard = document.createElement('div');
            fingerCard.className = `finger-card ${finger.state.toLowerCase()}`;

            fingerCard.innerHTML = `
                <div class="finger-info">
                    <span class="finger-name">${finger.name}</span>
                    <span class="finger-percent">${finger.fold_percent}%</span>
                </div>
                <div class="progress-track">
                    <div class="progress-fill" style="width: ${finger.fold_percent}%"></div>
                </div>
                <div class="finger-state-mini">${finger.state}</div>
            `;
            container.appendChild(fingerCard);
        });
    } else {
        container.innerHTML = '<p class="no-data">No hand detected</p>';
    }
}

// Poll every 100ms
setInterval(fetchStatus, 100);
