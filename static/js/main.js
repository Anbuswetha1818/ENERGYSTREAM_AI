document.addEventListener('DOMContentLoaded', () => {
    let trendChart, compChart;

    // Theme Colors
    const colors = {
        primary: '#6366f1',
        secondary: '#94a3b8',
        accent1: '#ec4899',
        accent2: '#06b6d4',
        grid: 'rgba(255, 255, 255, 0.05)',
        text: '#94a3b8'
    };

    const chartOptions = {
        responsive: true,
        maintainAspectRatio: false,
        interaction: { intersect: false, mode: 'index' },
        scales: {
            y: { grid: { color: colors.grid }, ticks: { color: colors.text } },
            x: { grid: { display: false }, ticks: { color: colors.text, maxTicksLimit: 10 } }
        },
        plugins: {
            legend: { position: 'top', labels: { color: '#f8fafc', font: { size: 12 } } }
        }
    };

    const fetchStats = async () => {
        // Support multiple ID variants across templates
        const avgEl = document.getElementById('avg-power') || document.getElementById('mean-power');
        const peakEl = document.getElementById('peak-power') || document.getElementById('max-power');
        const timeEl = document.getElementById('update-time') || document.getElementById('last-updated');

        if (!avgEl && !peakEl) return;

        try {
            const res = await fetch('/api/stats');
            const data = await res.json();
            if (avgEl) avgEl.innerText = `${data.mean_active_power.toFixed(2)} kW`;
            if (peakEl) peakEl.innerText = `${data.max_active_power.toFixed(2)} kW`;
            if (timeEl) {
                // Handle different time formats or elements
                const timeStr = data.last_updated.includes(' ') ? data.last_updated.split(' ')[1] : data.last_updated;
                timeEl.innerText = timeStr;
            }
        } catch (err) {
            console.error("Error fetching stats:", err);
        }
    };

    const fetchPredictions = async () => {
        const trendCanvas = document.getElementById('energyChart');
        const compCanvas = document.getElementById('comparisonChart');
        const refreshBtn = document.getElementById('refresh-btn');

        if (!trendCanvas && !compCanvas) return;

        if (refreshBtn) {
            refreshBtn.innerText = "Processing AI Models...";
            refreshBtn.disabled = true;
        }

        try {
            const res = await fetch('/api/predict');
            const data = await res.json();

            // Trend Chart (Home Page)
            if (trendCanvas) {
                const ctxTrend = trendCanvas.getContext('2d');
                if (trendChart) trendChart.destroy();
                trendChart = new Chart(ctxTrend, {
                    type: 'line',
                    data: {
                        labels: data.labels,
                        datasets: [
                            { label: 'Actual Load', data: data.actual, borderColor: colors.secondary, borderWidth: 2, pointRadius: 0, fill: false },
                            { label: 'LSTM Forecast', data: data.lstm, borderColor: colors.primary, borderWidth: 3, pointRadius: 0, fill: true, backgroundColor: 'rgba(99, 102, 241, 0.1)' },
                            { label: 'Random Forest', data: data.rf, borderColor: colors.accent1, borderWidth: 1, borderDash: [5, 5], pointRadius: 0, fill: false },
                            { label: 'Decision Tree', data: data.dt, borderColor: colors.accent2, borderWidth: 1, borderDash: [5, 5], pointRadius: 0, fill: false }
                        ]
                    },
                    options: chartOptions
                });
            }

            // Comparison Chart (Analysis Page)
            if (compCanvas) {
                const ctxComp = compCanvas.getContext('2d');
                if (compChart) compChart.destroy();

                // Update RMSE metrics text
                const lstmRmseEl = document.getElementById('lstm-rmse');
                const rfRmseEl = document.getElementById('rf-rmse');
                const dtRmseEl = document.getElementById('dt-rmse');

                if (lstmRmseEl) lstmRmseEl.innerText = `${data.metrics.lstm_rmse.toFixed(3)} kW`;
                if (rfRmseEl) rfRmseEl.innerText = `${data.metrics.rf_rmse.toFixed(3)} kW`;
                if (dtRmseEl) dtRmseEl.innerText = `${data.metrics.dt_rmse.toFixed(3)} kW`;

                compChart = new Chart(ctxComp, {
                    type: 'bar',
                    data: {
                        labels: ['LSTM (Proposed)', 'Random Forest', 'Decision Tree'],
                        datasets: [{
                            label: 'RMSE (lower is better)',
                            data: [data.metrics.lstm_rmse, data.metrics.rf_rmse, data.metrics.dt_rmse],
                            backgroundColor: [colors.primary, colors.accent1, colors.accent2],
                            borderRadius: 12
                        }]
                    },
                    options: {
                        ...chartOptions,
                        plugins: { ...chartOptions.plugins, legend: { display: false } }
                    }
                });
            }

        } catch (err) {
            console.error("Error fetching predictions:", err);
        } finally {
            if (refreshBtn) {
                refreshBtn.innerText = "Refresh Live Data";
                refreshBtn.disabled = false;
            }
        }
    };

    // Manual Prediction Logic (Forecast Page)
    const predictionForm = document.getElementById('prediction-form');
    if (predictionForm) {
        const loadSampleBtn = document.getElementById('load-sample-btn');
        const resultDiv = document.getElementById('prediction-result');
        const predictedValue = document.getElementById('predicted-value');

        if (loadSampleBtn) {
            loadSampleBtn.addEventListener('click', () => {
                console.log("Loading sample data into form...");
                const samples = {
                    "Voltage": 240.2,
                    "Global_intensity": 8.5,
                    "Global_reactive_power": 0.215
                };
                Object.keys(samples).forEach(key => {
                    const input = predictionForm.querySelector(`[name="${key}"]`);
                    if (input) {
                        input.value = samples[key];
                        input.style.backgroundColor = 'rgba(99, 102, 241, 0.1)'; // Highlight change
                        setTimeout(() => input.style.backgroundColor = '', 500);
                    }
                });
            });
        }

        predictionForm.addEventListener('submit', async (e) => {
            e.preventDefault();
            const btn = predictionForm.querySelector('button');
            const originalText = btn.innerText;
            btn.innerText = "Analyzing Grid Patterns...";
            btn.disabled = true;

            const formData = new FormData(predictionForm);
            const inputData = {};
            formData.forEach((value, key) => inputData[key] = parseFloat(value));

            try {
                const res = await fetch('/api/predict_manual', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(inputData)
                });
                const data = await res.json();
                if (data.success) {
                    predictedValue.innerText = `${data.prediction.toFixed(4)} ${data.unit}`;
                    resultDiv.classList.remove('hidden');
                    resultDiv.scrollIntoView({ behavior: 'smooth' });
                } else {
                    alert(`Error: ${data.error}`);
                }
            } catch (err) {
                console.error("Manual prediction error:", err);
                alert("Failed to connect to AI engine.");
            } finally {
                btn.innerText = originalText;
                btn.disabled = false;
            }
        });
    }

    // Refresh Hook
    const refreshBtn = document.getElementById('refresh-btn');
    if (refreshBtn) refreshBtn.addEventListener('click', fetchPredictions);

    // Initial Load
    fetchStats();
    fetchPredictions();
});
