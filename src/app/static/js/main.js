document.addEventListener('DOMContentLoaded', () => {
    const step1Enroll = document.getElementById('step-1-enroll');
    const step2Verify = document.getElementById('step-2-verify');
    const step2Info = document.getElementById('step-2-info');
    const activeUserDisplay = document.getElementById('active-user-display');
    const enrollmentStatus = document.getElementById('enrollment-status');

    const decisionText = document.getElementById('decision-text');
    const fakeScore = document.getElementById('fake-score');
    const matchScore = document.getElementById('match-score');
    const latency = document.getElementById('latency');

    const enrollUserInput = document.getElementById('enroll-user');
    const loginBtn = document.getElementById('login-btn');
    const enrollBtn = document.getElementById('enroll-btn');
    const restartBtn = document.getElementById('restart-btn');

    let isEnrolling = false;
    let currentUser = null;

    // Polling function
    setInterval(async () => {
        try {
            const response = await fetch('/api/status');
            const data = await response.json();
            
            if (isEnrolling) {
                if (data.decision.startsWith('ENROLLING')) {
                    enrollmentStatus.textContent = data.decision;
                }
                
                if (data.enrollment_done) {
                    // Transition to Verification!
                    isEnrolling = false;
                    enrollmentStatus.textContent = "Enrollment Complete!";
                    
                    // Tell backend to start verifying this user
                    await fetch('/api/set_user', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ user_id: currentUser })
                    });
                    
                    // Show Step 2 UI
                    step1Enroll.classList.add('hidden');
                    step2Verify.classList.remove('hidden');
                    step2Info.classList.remove('hidden');
                    activeUserDisplay.textContent = currentUser;
                }
            } else {
                // Update Verification UI Texts
                decisionText.textContent = data.decision.replace('_', ' ');
                fakeScore.textContent = data.fake_score.toFixed(3);
                matchScore.textContent = data.match_score.toFixed(3);
                latency.textContent = data.latency_ms.toFixed(1) + ' ms';

                // Update Verification CSS Styles
                step2Verify.className = 'status-overlay'; // reset
                if (data.decision === 'ACCEPT') {
                    step2Verify.classList.add('state-accept');
                } else if (data.decision === 'REJECT_FAKE') {
                    step2Verify.classList.add('state-reject-fake');
                } else if (data.decision === 'REJECT_IDENTITY') {
                    step2Verify.classList.add('state-reject-identity');
                } else {
                    step2Verify.classList.add('state-waiting');
                }
            }
            
        } catch (error) {
            console.error("Error fetching status:", error);
        }
    }, 250);

    // Start Enrollment
    loginBtn.addEventListener('click', async () => {
        const user_id = enrollUserInput.value.trim();
        if (!user_id) {
            alert("Please enter your name to login!");
            return;
        }
        
        currentUser = user_id;
        
        // Set the active user on the server
        await fetch('/api/set_user', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ user_id: currentUser })
        });
        
        // Skip enrollment UI and jump straight to verification!
        step1Enroll.classList.add('hidden');
        step2Verify.classList.remove('hidden');
        step2Info.classList.remove('hidden');
        activeUserDisplay.textContent = currentUser;
    });

    enrollBtn.addEventListener('click', async () => {
        const user_id = enrollUserInput.value.trim();
        if (!user_id) {
            alert("Please enter a name first!");
            return;
        }

        currentUser = user_id;
        isEnrolling = true;
        enrollBtn.textContent = "Scanning...";
        enrollBtn.disabled = true;
        enrollmentStatus.textContent = "Looking for face...";

        try {
            await fetch('/api/enroll', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ user_id })
            });
        } catch (error) {
            alert(`Error: ${error}`);
            isEnrolling = false;
            enrollBtn.textContent = "Start Enrollment";
            enrollBtn.disabled = false;
        }
    });

    // Reset Flow
    restartBtn.addEventListener('click', async () => {
        // Unset target
        await fetch('/api/set_user', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ user_id: "" })
        });
        
        currentUser = null;
        enrollUserInput.value = "";
        enrollBtn.textContent = "Start Enrollment";
        enrollBtn.disabled = false;
        enrollmentStatus.textContent = "";
        
        // Transition back to Step 1
        step2Verify.classList.add('hidden');
        step2Info.classList.add('hidden');
        step1Enroll.classList.remove('hidden');
    });
});
