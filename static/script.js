document.addEventListener('DOMContentLoaded', () => {
  const form = document.getElementById('weather-form');
  const predictBtn = document.getElementById('predict-btn');
  const resetBtn = document.getElementById('reset-btn');
  const message = document.getElementById('form-message');
  const placeholder = document.getElementById('result-placeholder');
  const predictionPanel = document.getElementById('result-prediction');
  const predictionLoading = document.getElementById('prediction-loading');
  const resultCard = document.querySelector('.result-card');
  const themeButtons = document.querySelectorAll('.theme-btn');
  const historyList = document.getElementById('history-list');
  const clearHistoryBtn = document.getElementById('clear-history');
  const fields = ['temperature','humidity','wind_speed','wind_direction','cloud_cover','pressure','visibility','precipitation','uv_index','dew_point'];
  const ranges = {
    temperature: [-20, 50], humidity: [0, 100], wind_speed: [0, 120], wind_direction: [0, 360],
    cloud_cover: [0, 100], pressure: [950, 1050], visibility: [0, 30], precipitation: [0, 50],
    uv_index: [0, 14], dew_point: [-20, 40]
  };

  const setTheme = (theme) => {
    document.documentElement.dataset.theme = theme;
    localStorage.setItem('weatherpredict-theme', theme);
    themeButtons.forEach(btn => btn.classList.toggle('selected', btn.dataset.theme === theme));
  };
  const savedTheme = localStorage.getItem('weatherpredict-theme');
  setTheme(savedTheme === 'light' || savedTheme === 'dark' ? savedTheme : 'dark');
  themeButtons.forEach(btn => btn.addEventListener('click', () => setTheme(btn.dataset.theme)));

  // Convert the numeric controls into slider-style controls while retaining a visible value.
  fields.forEach(id => {
    const input = document.getElementById(id);
    if (!input) return;
    const initialValue = input.value || input.placeholder || ranges[id][0];
    input.type = 'range';
    input.value = initialValue;
    input.min = ranges[id][0];
    input.max = ranges[id][1];
    input.step = id === 'wind_direction' || id === 'pressure' ? '1' : '0.1';
    const valueBox = document.createElement('output');
    valueBox.className = 'range-value';
    valueBox.setAttribute('for', id);
    valueBox.textContent = input.value || input.placeholder || ranges[id][0];
    input.parentElement.classList.add('range-input-group');
    input.parentElement.insertBefore(valueBox, input.nextSibling);
    const update = () => {
      valueBox.textContent = input.value;
      const min = Number(input.min), max = Number(input.max), value = Number(input.value);
      const percent = max > min ? ((value - min) / (max - min)) * 100 : 0;
      input.style.background = `linear-gradient(90deg, var(--accent) 0%, var(--accent) ${percent}%, var(--input) ${percent}%, var(--input) 100%)`;
    };
    input.addEventListener('input', update);
    update();
  });

  const showMessage = (text = '', type = '') => { message.textContent = text; message.className = `form-message ${type}`; };

  const presetValues = {
    Sunny: { temperature:30, humidity:35, wind_speed:10, wind_direction:180, cloud_cover:10, pressure:1018, visibility:15, precipitation:0, uv_index:8, dew_point:12 },
    'Partly Cloudy': { temperature:25, humidity:50, wind_speed:15, wind_direction:180, cloud_cover:40, pressure:1015, visibility:12, precipitation:1, uv_index:5, dew_point:14 },
    Cloudy: { temperature:18, humidity:65, wind_speed:18, wind_direction:180, cloud_cover:75, pressure:1010, visibility:8, precipitation:2, uv_index:3, dew_point:13 },
    Rainy: { temperature:15, humidity:80, wind_speed:25, wind_direction:180, cloud_cover:90, pressure:1005, visibility:6, precipitation:8, uv_index:2, dew_point:12 },
    Thunderstorm: { temperature:22, humidity:85, wind_speed:45, wind_direction:180, cloud_cover:95, pressure:998, visibility:4, precipitation:15, uv_index:1.5, dew_point:18 },
    Foggy: { temperature:10, humidity:92, wind_speed:5, wind_direction:180, cloud_cover:70, pressure:1012, visibility:2, precipitation:0.5, uv_index:2, dew_point:9 },
    'Hot Sunny': { temperature:38, humidity:28, wind_speed:8, wind_direction:90, cloud_cover:5, pressure:1022, visibility:20, precipitation:0, uv_index:11, dew_point:13 },
    'Light Rain': { temperature:17, humidity:78, wind_speed:18, wind_direction:225, cloud_cover:82, pressure:1008, visibility:8, precipitation:4, uv_index:2, dew_point:13 },
    'Severe Storm': { temperature:23, humidity:90, wind_speed:65, wind_direction:270, cloud_cover:98, pressure:992, visibility:3, precipitation:28, uv_index:1, dew_point:21 },
    'Dense Fog': { temperature:7, humidity:97, wind_speed:3, wind_direction:45, cloud_cover:88, pressure:1014, visibility:0.8, precipitation:0.2, uv_index:1, dew_point:7 }
  };

  document.querySelectorAll('.preset-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      const values = presetValues[btn.dataset.preset];
      if (!values) return;
      fields.forEach(id => {
        const input = document.getElementById(id);
        input.value = values[id];
        input.dispatchEvent(new Event('input', { bubbles: true }));
      });
      showMessage(`${btn.dataset.preset} preset applied. Click Predict Weather to test it.`, 'success');
      document.getElementById('predict').scrollIntoView({ behavior: 'smooth', block: 'start' });
    });
  });

  form.addEventListener('submit', async (event) => {
    event.preventDefault();
    showMessage('');
    const payload = {};
    for (const id of fields) {
      const input = document.getElementById(id);
      if (input.value === '') { input.focus(); showMessage(`Please enter ${input.labels[0].textContent.trim()}.`, 'error'); return; }
      const value = Number(input.value);
      if (!Number.isFinite(value)) { input.focus(); showMessage('Please enter valid numbers in all fields.', 'error'); return; }
      payload[id] = value;
    }

    predictBtn.disabled = true;
    predictBtn.innerHTML = '⏳ Predicting...';
    placeholder.hidden = true;
    if (window.matchMedia('(max-width: 700px)').matches && resultCard) {
      resultCard.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
    predictionPanel.hidden = true;
    if (predictionLoading) predictionLoading.hidden = false;
    try {
      const response = await fetch('/api/predict', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(payload) });
      const data = await response.json();
      if (!response.ok || !data.success) throw new Error(data.error || 'Prediction failed.');
      displayPrediction(data);
      addHistory(data, payload);
      showMessage('Prediction completed successfully.', 'success');
    } catch (error) {
      console.error(error);
      showMessage(error.message || 'Unable to connect to the prediction service.', 'error');
    } finally {
      if (predictionLoading) predictionLoading.hidden = true;
      predictBtn.disabled = false;
      predictBtn.innerHTML = '✦ Predict Weather <span>→</span>';
    }
  });

  function displayPrediction(data) {
    placeholder.hidden = true;
    predictionPanel.hidden = false;
    document.getElementById('prediction-icon').textContent = data.icon;
    document.getElementById('prediction-value').textContent = data.prediction;
    const confidenceValue = Math.min(100, Math.max(0, Number(data.confidence) || 0));
    document.getElementById('confidence-value').textContent = `${confidenceValue}%`;
    animateConfidence(confidenceValue);
    document.getElementById('prediction-description').textContent = data.description;
    document.getElementById('prediction-timestamp').textContent = `Predicted at ${data.timestamp}`;
    const labels = {temperature:'Temperature', humidity:'Humidity', wind_speed:'Wind Speed', wind_direction:'Wind Direction', cloud_cover:'Cloud Cover', pressure:'Pressure', visibility:'Visibility', precipitation:'Precipitation', uv_index:'UV Index', dew_point:'Dew Point'};
    const units = {temperature:'°C', humidity:'%', wind_speed:'km/h', wind_direction:'°', cloud_cover:'%', pressure:'hPa', visibility:'km', precipitation:'mm', uv_index:'', dew_point:'°C'};
    const summary = document.getElementById('summary-grid');
    summary.replaceChildren();
    Object.entries(data.input_summary).forEach(([key, value]) => {
      const item = document.createElement('div'); item.className = 'summary-item';
      const label = document.createElement('span'); label.textContent = labels[key] || key;
      const val = document.createElement('strong'); val.textContent = `${value}${units[key] ? ` ${units[key]}` : ''}`;
      item.append(label, val); summary.appendChild(item);
    });
  }

  function animateConfidence(target) {
    const fill = document.getElementById('confidence-fill');
    if (!fill) return;
    fill.style.width = '0%';
    fill.classList.remove('confidence-animating');
    // Start from zero and smoothly grow to the model's actual confidence.
    requestAnimationFrame(() => {
      requestAnimationFrame(() => {
        fill.classList.add('confidence-animating');
        fill.style.width = `${target}%`;
      });
    });
  }

  function addHistory(data, payload) {
    const history = JSON.parse(localStorage.getItem('weatherpredict-history') || '[]');
    history.unshift({ prediction: data.prediction, icon: data.icon, confidence: data.confidence, timestamp: data.timestamp, values: payload });
    history.splice(8);
    localStorage.setItem('weatherpredict-history', JSON.stringify(history));
    renderHistory();
  }

  function renderHistory() {
    if (!historyList) return;
    const history = JSON.parse(localStorage.getItem('weatherpredict-history') || '[]');
    historyList.replaceChildren();
    if (!history.length) {
      historyList.innerHTML = '<p class="history-empty">Your recent predictions will appear here.</p>';
      return;
    }
    history.forEach(item => {
      const row = document.createElement('button');
      row.type = 'button';
      row.className = 'history-item';
      row.innerHTML = `<span class="history-icon">${item.icon}</span><span class="history-main"><strong>${item.prediction}</strong><small>${item.timestamp}</small></span><strong class="history-confidence">${item.confidence}%</strong>`;
      row.addEventListener('click', () => {
        fields.forEach(id => {
          const input = document.getElementById(id);
          if (item.values[id] !== undefined) { input.value = item.values[id]; input.dispatchEvent(new Event('input', { bubbles: true })); }
        });
        document.getElementById('predict').scrollIntoView({ behavior: 'smooth', block: 'start' });
        showMessage('Previous prediction values restored.', 'success');
      });
      historyList.appendChild(row);
    });
  }

  clearHistoryBtn?.addEventListener('click', () => {
    localStorage.removeItem('weatherpredict-history');
    renderHistory();
  });

  resetBtn.addEventListener('click', () => {
    form.reset();
    fields.forEach(id => document.getElementById(id)?.dispatchEvent(new Event('input', { bubbles: true })));
    placeholder.hidden = false;
    predictionPanel.hidden = true;
    if (predictionLoading) predictionLoading.hidden = true;
    document.getElementById('confidence-fill').style.width = '0%';
    showMessage('');
  });

  async function loadMetrics() {
    try {
      const response = await fetch('/api/metrics'); const data = await response.json();
      if (!response.ok || !data.success) throw new Error(data.error || 'Metrics unavailable.');
      const m = data.metrics;
      document.getElementById('metric-accuracy').textContent = `${m.accuracy}%`;
      document.getElementById('metric-precision').textContent = `${m.precision}%`;
      document.getElementById('metric-recall').textContent = `${m.recall}%`;
      document.getElementById('metric-f1').textContent = `${m.f1_score}%`;
    } catch (e) { console.error('Metrics:', e); }
  }

  async function loadDatasetInfo() {
    try {
      const response = await fetch('/api/dataset'); const data = await response.json();
      if (!response.ok || !data.success) throw new Error(data.error || 'Dataset information unavailable.');
      document.getElementById('dataset-records').textContent = Number(data.total_records).toLocaleString();
      document.getElementById('dataset-features').textContent = data.num_features;
      document.getElementById('dataset-class-count').textContent = data.num_classes;
      document.getElementById('dataset-classes').textContent = data.classes.join(', ');
    } catch (e) { console.error('Dataset:', e); }
  }

  // Navigation buttons scroll to their actual sections and update the active state.
  document.querySelectorAll('.nav-links a[href^="#"]').forEach(link => {
    link.addEventListener('click', () => {
      document.querySelectorAll('.nav-links a').forEach(a => a.classList.remove('active'));
      link.classList.add('active');
    });
  });

  renderHistory();
  loadMetrics();
  loadDatasetInfo();
});
