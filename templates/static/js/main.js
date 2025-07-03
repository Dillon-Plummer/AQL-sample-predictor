document.addEventListener('DOMContentLoaded', function () {

    // --- DOM Element Cache ---
    const elements = {
        dataForm: document.getElementById('data-form'),
        csvFileInput: document.getElementById('csvFile'),
        demoDataBtn: document.getElementById('demoDataBtn'),
        dataStatus: document.getElementById('data-status'),
        errorBanner: document.getElementById('error-banner'),
        errorMessage: document.getElementById('error-message'),

        stepCards: {
            step1: document.getElementById('step1'),
            step2: document.getElementById('step2'),
            step3: document.getElementById('step3'),
            step4: document.getElementById('step4'),
            step5: document.getElementById('step5'),
        },

        runButtons: {
            step1: document.getElementById('runStep1'),
            step2: document.getElementById('runStep2'),
            step3: document.getElementById('runStep3'),
            step4: document.getElementById('runStep4'),
            step5: document.getElementById('runStep5'),
        },

        resultsContainers: {
            step1: document.getElementById('step1-results'),
            step2: document.getElementById('step2-results'),
            step3: document.getElementById('step3-results'),
            step4: document.getElementById('step4-results'),
            step5: document.getElementById('step5-results'),
        },

        // Scenario comparison elements
        scenarioSection: document.getElementById('scenario-comparison-section'),
        scenarioTableBody: document.getElementById('scenario-table-body'),
        addToComparisonBtn: document.getElementById('add-to-comparison'),
        addToComparisonContainer: document.getElementById('add-to-comparison-container'),
        clearScenariosBtn: document.getElementById('clear-scenarios'),
    };

    // --- State Management ---
    let state = {
        dataSourceName: '',
        scenarios: [],
        lotSize: 0,
    };

    // --- Utility Functions ---
    function toggleSpinner(button, show) {
        const spinner = button.querySelector('.spinner');
        if (spinner) {
            spinner.classList.toggle('hidden', !show);
            button.disabled = show;
        }
    }

    function showError(message) {
        elements.errorMessage.textContent = message;
        elements.errorBanner.classList.remove('hidden');
    }

    function hideError() {
        elements.errorBanner.classList.add('hidden');
    }

    function enableStep(stepNumber) {
        const stepCard = elements.stepCards[`step${stepNumber}`];
        if (stepCard) {
            stepCard.classList.remove('disabled');
        }
    }

    async function handleFetch(url, options, button) {
        hideError();
        toggleSpinner(button, true);
        try {
            const response = await fetch(url, options);
            const result = await response.json();
            if (!response.ok) {
                throw new Error(result.message || `HTTP error! status: ${response.status}`);
            }
            return result;
        } catch (error) {
            console.error('Fetch Error:', error);
            showError(error.message);
            return null;
        } finally {
            toggleSpinner(button, false);
        }
    }

    // --- Event Listeners ---

    // Data Loading
    elements.demoDataBtn.addEventListener('click', () => loadData('demo'));
    elements.csvFileInput.addEventListener('change', () => loadData('upload'));

    async function loadData(source) {
        const formData = new FormData(elements.dataForm);
        formData.append('source', source);

        const result = await handleFetch('/upload_data', { method: 'POST', body: formData }, elements.demoDataBtn);

        if (result && result.status === 'success') {
            elements.dataStatus.textContent = result.message;
            state.dataSourceName = result.filename;
            enableStep(1);
        }
    }

    // Accordions
    document.querySelectorAll('.accordion-toggle').forEach(button => {
        button.addEventListener('click', () => {
            const content = button.nextElementSibling;
            content.classList.toggle('open');
        });
    });

    // Step 1: Bayesian Model
    elements.runButtons.step1.addEventListener('click', async () => {
        const payload = {
            prior_mean: document.getElementById('prior-mean').value,
            prior_sigma: document.getElementById('prior-std').value,
        };
        const result = await handleFetch('/run_step_1', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        }, elements.runButtons.step1);

        if (result) {
            document.getElementById('posterior-summary').innerHTML = result.summary_html;
            document.getElementById('posterior-plot').src = 'data:image/png;base64,' + result.posterior_plot;
            document.getElementById('trace-plot').src = 'data:image/png;base64,' + result.trace_plot;
            elements.resultsContainers.step1.classList.remove('hidden');
            enableStep(2);
        }
    });

    // Step 2: Linear Regression
    elements.runButtons.step2.addEventListener('click', async () => {
        const result = await handleFetch('/run_step_2', { method: 'POST' }, elements.runButtons.step2);
        if (result) {
            document.getElementById('regression-summary').innerHTML = result.summary_html;
            document.getElementById('qq-plot').src = 'data:image/png;base64,' + result.qq_plot;
            document.getElementById('shapiro-stat').textContent = result.shapiro_stat;
            document.getElementById('shapiro-p').textContent = result.shapiro_p;
            document.getElementById('bp-stat').textContent = result.bp_stat;
            document.getElementById('bp-p').textContent = result.bp_p;
            elements.resultsContainers.step2.classList.remove('hidden');
            enableStep(3);
        }
    });

    // Step 3: Hypergeometric Model
    elements.runButtons.step3.addEventListener('click', async () => {
        const result = await handleFetch('/run_step_3', { method: 'POST' }, elements.runButtons.step3);
        if (result) {
            document.getElementById('risk-score').textContent = result.risk_score;
            elements.resultsContainers.step3.classList.remove('hidden');
            enableStep(4);
        }
    });

    // Step 4: Random Forest
    elements.runButtons.step4.addEventListener('click', async () => {
        const result = await handleFetch('/run_step_4', { method: 'POST' }, elements.runButtons.step4);
        if (result) {
            document.getElementById('final-aql').textContent = result.final_aql;
            document.getElementById('rf-cv-score').textContent = result.cv_score;
            document.getElementById('importance-plot').src = 'data:image/png;base64,' + result.importance_plot;
            elements.resultsContainers.step4.classList.remove('hidden');
            elements.addToComparisonContainer.classList.remove('hidden');
            enableStep(5);
        }
    });

    // Step 5: Cost-Benefit Analysis
    elements.runButtons.step5.addEventListener('click', () => {
        const finalAQL = document.getElementById('final-aql').textContent;
        const costPerInspection = parseFloat(document.getElementById('cost-per-inspection').value);
        const costPerDefect = parseFloat(document.getElementById('cost-per-defect').value);
        const riskScore = parseFloat(document.getElementById('risk-score').textContent);

        if (isNaN(costPerInspection) || isNaN(costPerDefect) || !finalAQL) return;

        // Simplified sample size lookup based on AQL
        let sampleSize = 80; // default for AQL 1.5
        if (finalAQL === "1.0") sampleSize = 125;
        if (finalAQL === "0.65") sampleSize = 200;
        if (finalAQL === "2.5") sampleSize = 50;

        const inspectionCost = sampleSize * costPerInspection;
        const riskCost = (riskScore / 100) * costPerDefect;

        document.getElementById('inspection-cost').textContent = `$${inspectionCost.toFixed(2)}`;
        document.getElementById('risk-cost').textContent = `$${riskCost.toFixed(2)}`;
        elements.resultsContainers.step5.classList.remove('hidden');
    });

    // Scenario Comparison Logic
    elements.addToComparisonBtn.addEventListener('click', () => {
        const scenario = {
            id: state.scenarios.length + 1,
            dataSource: state.dataSourceName,
            priorMean: document.getElementById('prior-mean').value,
            priorStd: document.getElementById('prior-std').value,
            ris                                                                                                                                                                                kScore: document.getElementById('risk-score').textContent,
            finalAQL: document.getElementById('final-aql').textContent,
        };
        state.scenarios.push(scenario);
        renderScenarioTable();
        elements.scenarioSection.classList.remove('hidden');
    });

    function renderScenarioTable() {
        elements.scenarioTableBody.innerHTML = '';
        state.scenarios.forEach(s => {
            const row = document.createElement('tr');
            row.innerHTML = `
                <td class="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">${s.id}</td>
                <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500">${s.dataSource}</td>
                <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500">(${s.priorMean}, ${s.priorStd})</td>
                <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500">${s.riskScore}</td>
                <td class="px-6 py-4 whitespace-nowrap text-sm font-bold text-indigo-600">${s.finalAQL}</td>
            `;
            elements.scenarioTableBody.appendChild(row);
        });
    }

    elements.clearScenariosBtn.addEventListener('click', () => {
        state.scenarios = [];
        renderScenarioTable();
        elements.scenarioSection.classList.add('hidden');
    });
});
