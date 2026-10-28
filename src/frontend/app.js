// Vanilla JS, no build step. Leaflet is the one external library this page
// loads (via CDN, in index.html) - unavoidable for a real interactive map;
// everything else here calls only the FastAPI backend at relative URLs
// (same origin, works identically on localhost and after deployment).

const statusBox = document.getElementById("statusBox");
const content = document.getElementById("content");
const searchForm = document.getElementById("searchForm");
const searchInput = document.getElementById("searchInput");
const suggestionsEl = document.getElementById("suggestions");
const locateBtn = document.getElementById("locateBtn");

let leafletMap = null;
let floodMap = null;
let activeSuggestionIndex = -1;
let currentSuggestions = [];

// Monotonic request guard: every load gets a ticket; only the response
// matching the CURRENT (latest) ticket is ever allowed to touch the DOM.
// Without this, an older slow request can resolve after a newer fast one
// and silently overwrite it - the "stuck on Loading..." / stale-content
// bug. This makes that entire class of bug impossible regardless of
// network timing.
let currentRequestId = 0;

function iconFor(condition) {
  const c = condition.toLowerCase();
  if (c.includes("thunder")) return "⛈️";
  if (c.includes("snow")) return "❄️";
  if (c.includes("rain") || c.includes("drizzle") || c.includes("shower")) return "\u{1F327}️";
  if (c.includes("fog")) return "\u{1F32B}️";
  if (c.includes("overcast")) return "☁️";
  if (c.includes("partly") || c.includes("cloud")) return "⛅";
  if (c.includes("clear")) return "☀️";
  return "\u{1F324}️";
}

function setStatus(text, isError) {
  statusBox.hidden = false;
  content.hidden = true;
  statusBox.textContent = text;
  statusBox.className = "status-box" + (isError ? " error-text" : "");
}

function fmtHour(iso) { return iso.slice(11, 16); }
function fmtClock(iso) {
  if (!iso) return "--:--";
  const [, time] = iso.split("T");
  return time || "--:--";
}
function fmtDay(iso, index) {
  if (index === 0) return "Today";
  const d = new Date(iso + "T00:00:00");
  return d.toLocaleDateString(undefined, { weekday: "short" });
}

// ---------- UV / AQI helpers ----------

function uvCategory(uv) {
  if (uv == null) return { text: "—", color: "var(--low)", pct: 0 };
  if (uv < 3) return { text: "Low", color: "var(--low)", pct: (uv / 12) * 100 };
  if (uv < 6) return { text: "Moderate", color: "var(--moderate)", pct: (uv / 12) * 100 };
  if (uv < 8) return { text: "High", color: "#ff9a3d", pct: (uv / 12) * 100 };
  if (uv < 11) return { text: "Very High", color: "var(--high)", pct: (uv / 12) * 100 };
  return { text: "Extreme", color: "#c04dff", pct: 100 };
}

function aqiColor(category) {
  if (category === "Good" || category === "Moderate") return "var(--low)";
  if (category === "Unhealthy for Sensitive Groups") return "var(--moderate)";
  if (category === "Unhealthy") return "#ff9a3d";
  return "var(--high)";
}

// ---------- Map ----------

function initMap(lat, lon, label) {
  const mapEl = document.getElementById("map");
  try {
    if (leafletMap) { leafletMap.remove(); leafletMap = null; }
    leafletMap = L.map(mapEl, { zoomControl: true }).setView([lat, lon], 7);

    L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
      maxZoom: 12,
      attribution: "&copy; OpenStreetMap contributors",
    }).addTo(leafletMap);

    L.marker([lat, lon]).addTo(leafletMap).bindPopup(label);

    // Live precipitation radar overlay (RainViewer, free public API) -
    // optional: if it fails to load, the base map above still works fine
    // without it, so a failure here is silent rather than shown in the UI.
    fetch("https://api.rainviewer.com/public/weather-maps.json")
      .then((r) => r.json())
      .then((data) => {
        const frames = (data.radar && data.radar.past) || [];
        if (!frames.length) return;
        const latest = frames[frames.length - 1];
        const tileUrl = `${data.host}${latest.path}/256/{z}/{x}/{y}/2/1_1.png`;
        L.tileLayer(tileUrl, { opacity: 0.6, maxZoom: 12 }).addTo(leafletMap);
      })
      .catch(() => {});
  } catch (err) {
    mapEl.innerHTML = '<div style="padding:16px;color:rgba(255,255,255,0.6);font-size:13px;">Map unavailable right now.</div>';
  }
}

function talukColor(band) {
  if (band === "LOW") return "#3ddc84";
  if (band === "MODERATE") return "#f7c948";
  return "#ff5c5c";
}

async function initFloodTalukMap() {
  const mapEl = document.getElementById("floodMap");
  const legendEl = document.getElementById("talukLegend");
  try {
    const resp = await fetch("/predict/taluk/today");
    if (!resp.ok) throw new Error("request failed");
    const data = await resp.json();

    if (floodMap) { floodMap.remove(); floodMap = null; }
    floodMap = L.map(mapEl, { zoomControl: false, dragging: true, scrollWheelZoom: false })
      .setView([12.46, 75.78], 9);
    L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", { maxZoom: 11 }).addTo(floodMap);

    // Approximate taluk headquarters coordinates (same ones used by the
    // backend's Voronoi split) - the API doesn't return lat/lon per taluk,
    // so they're reproduced here just for marker placement.
    const taluCoords = {
      Madikeri: [12.4244, 75.7382],
      Virajpet: [12.1936, 75.8064],
      Somwarpet: [12.5985, 75.8517],
    };

    legendEl.innerHTML = "";
    data.taluks.forEach((t) => {
      const coords = taluCoords[t.taluk];
      if (!coords) return;
      const color = talukColor(t.risk_band);
      L.circleMarker(coords, {
        radius: 16,
        color: "#fff",
        weight: 2,
        fillColor: color,
        fillOpacity: 0.85,
      })
        .addTo(floodMap)
        .bindPopup(`<b>${t.taluk}</b><br>${t.risk_band} (${Math.round(t.flood_risk_probability * 100)}%)`);

      const legendItem = document.createElement("div");
      legendItem.innerHTML = `<span class="dot" style="background:${color}"></span>${t.taluk}: ${t.risk_band} (${Math.round(t.flood_risk_probability * 100)}%)`;
      legendEl.appendChild(legendItem);
    });
  } catch (err) {
    mapEl.innerHTML = '<div style="padding:12px;color:rgba(255,255,255,0.5);font-size:12px;">Per-taluk map unavailable right now.</div>';
  }
}

// ---------- 10-day outlook chart (click a day to see its details) ----------

let outlookDaysData = [];
let selectedOutlookIndex = 0;

async function initOutlookChart() {
  const svg = document.getElementById("outlookChart");
  const detail = document.getElementById("outlookDetail");
  try {
    const resp = await fetch("/predict/outlook");
    if (!resp.ok) throw new Error("request failed");
    const data = await resp.json();
    outlookDaysData = data.days;
    selectedOutlookIndex = 0;
    renderOutlookChart(svg, outlookDaysData);
    showOutlookDay(0);
  } catch (err) {
    svg.innerHTML = "";
    detail.innerHTML = '<span style="color:var(--text-faint)">Outlook unavailable right now.</span>';
  }
}

function renderOutlookChart(svg, days) {
  const W = 320, H = 90, padTop = 6, padBottom = 6;
  const maxH = H - padTop - padBottom;
  const n = days.length;
  const slot = W / n;
  const barW = slot * 0.62;
  const baseline = padTop + maxH;

  let parts = `<line x1="0" y1="${baseline}" x2="${W}" y2="${baseline}" stroke="rgba(255,255,255,0.15)" stroke-width="1"/>`;
  days.forEach((d, i) => {
    const x = i * slot + (slot - barW) / 2;
    const h = Math.max(3, d.flood_risk_probability * maxH);
    const y = baseline - h;
    const color = talukColor(d.risk_band);
    const opacity = d.is_live_input ? 1 : 0.55;
    const label = i === 0 ? "Today" : `+${i}d`;
    parts += `<rect data-index="${i}" x="${x.toFixed(1)}" y="${y.toFixed(1)}" width="${barW.toFixed(1)}" height="${h.toFixed(1)}" rx="3" fill="${color}" opacity="${opacity}"><title>${label} (${d.date}): ${Math.round(d.flood_risk_probability * 100)}% ${d.risk_band}${d.is_live_input ? "" : " (trend-projected)"} - click for details</title></rect>`;
  });
  svg.innerHTML = parts;

  svg.querySelectorAll("rect[data-index]").forEach((rect) => {
    rect.addEventListener("click", () => showOutlookDay(Number(rect.dataset.index)));
  });
}

function showOutlookDay(index) {
  const d = outlookDaysData[index];
  if (!d) return;
  selectedOutlookIndex = index;

  const svg = document.getElementById("outlookChart");
  svg.querySelectorAll("rect[data-index]").forEach((rect) => {
    rect.setAttribute("stroke", Number(rect.dataset.index) === index ? "#fff" : "none");
    rect.setAttribute("stroke-width", "2");
  });

  const dateLabel = new Date(d.date + "T00:00:00").toLocaleDateString(undefined, {
    weekday: "long", month: "short", day: "numeric",
  });
  const whenLabel = index === 0 ? "Today" : `In ${index} day${index > 1 ? "s" : ""}`;
  const pct = Math.round(d.flood_risk_probability * 100);
  const sourceNote = d.is_live_input ? "based on a live weather forecast" : "trend-projected (beyond the live forecast window)";

  document.getElementById("outlookDetail").innerHTML = `
    <div><span class="od-date">${whenLabel} · ${dateLabel}</span></div>
    <div><span class="od-band ${d.risk_band}">${d.risk_band} RISK</span> &nbsp; ${pct}% probability</div>
    <div class="od-source">${sourceNote}</div>
  `;
}

// ---------- Seasonal climatology chart (76 years of real data) ----------

async function initClimatologyChart() {
  const svg = document.getElementById("climatologyChart");
  const caption = document.getElementById("climatologyCaption");
  try {
    const resp = await fetch("/predict/climatology");
    if (!resp.ok) throw new Error("request failed");
    const data = await resp.json();
    renderClimatologyChart(svg, data.months);

    const sorted = [...data.months].sort((a, b) => b.flood_risk_rate_pct - a.flood_risk_rate_pct);
    const riskiest = sorted.slice(0, 2).map((m) => m.month).join("-");
    const safest = sorted[sorted.length - 1].month;
    caption.textContent =
      `Based on ${data.years_of_data} years of real observed data: ${riskiest} are historically Kodagu's ` +
      `highest flood-risk months, ${safest} the lowest. This is a seasonal pattern, not a day-specific forecast.`;
  } catch (err) {
    svg.innerHTML = "";
    caption.textContent = "Seasonal pattern unavailable right now.";
  }
}

function renderClimatologyChart(svg, months) {
  const W = 640, H = 120, padTop = 6, padBottom = 20;
  const maxH = H - padTop - padBottom;
  const n = months.length;
  const slot = W / n;
  const barW = slot * 0.6;
  const baseline = padTop + maxH;
  const maxRate = Math.max(...months.map((m) => m.flood_risk_rate_pct), 1);
  const currentMonth = new Date().getMonth();

  let parts = `<line x1="0" y1="${baseline}" x2="${W}" y2="${baseline}" stroke="rgba(255,255,255,0.15)" stroke-width="1"/>`;
  months.forEach((m, i) => {
    const x = i * slot + (slot - barW) / 2;
    const h = Math.max(2, (m.flood_risk_rate_pct / maxRate) * maxH);
    const y = baseline - h;
    const isCurrent = i === currentMonth;
    const color = m.flood_risk_rate_pct >= maxRate * 0.66 ? "#ff5c5c" : m.flood_risk_rate_pct >= maxRate * 0.33 ? "#f7c948" : "#3ddc84";
    parts += `<rect x="${x.toFixed(1)}" y="${y.toFixed(1)}" width="${barW.toFixed(1)}" height="${h.toFixed(1)}" rx="3" fill="${color}" opacity="${isCurrent ? 1 : 0.7}" stroke="${isCurrent ? "#fff" : "none"}" stroke-width="1.5"><title>${m.month}: ${m.flood_risk_rate_pct}% of days were flood-risk days (${m.n_days} days across all years)</title></rect>`;
    parts += `<text x="${(x + barW / 2).toFixed(1)}" y="${H - 6}" font-size="10" fill="${isCurrent ? "var(--text)" : "var(--text-faint)"}" text-anchor="middle">${m.month}</text>`;
  });
  svg.innerHTML = parts;
}

// ---------- Historical date picker ----------

const historyForm = document.getElementById("historyForm");
const historyResult = document.getElementById("historyResult");

document.getElementById("historyDate").max = new Date().toISOString().slice(0, 10);

historyForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  const date = document.getElementById("historyDate").value;
  if (!date) return;
  historyResult.innerHTML = "Checking…";
  try {
    const resp = await fetch(`/predict/historical/${date}`);
    const body = await resp.json();
    if (!resp.ok) {
      historyResult.innerHTML = `<div class="h-error">${body.detail || "No data for that date."}</div>`;
      return;
    }
    const pct = Math.round(body.flood_risk_probability * 100);
    const actualText = body.actual_outcome === "flood_risk_day" ? "flood-risk day" : "normal day";
    const matchText = body.model_matched_actual ? "model matched the real outcome ✓" : "model missed the real outcome ✗";
    const why = body.explanation
      .map((f) => `<span class="${f.direction}">${f.direction === "up" ? "↑" : "↓"} ${f.feature}</span>`)
      .join(" &nbsp; ");
    historyResult.innerHTML = `
      <div class="h-row">
        <span class="h-band ${body.risk_band}">${body.risk_band} RISK</span>
        <span>${pct}%</span>
        <span class="h-match">Actual: ${actualText} — ${matchText}</span>
      </div>
      <div class="h-why">${why}</div>
    `;
  } catch (err) {
    historyResult.innerHTML = '<div class="h-error">Couldn\'t check that date right now.</div>';
  }
});

// ---------- Render ----------

function render(data) {
  statusBox.hidden = true;
  content.hidden = false;

  const loc = data.location;
  const nameParts = [loc.name];
  if (loc.admin1) nameParts.push(loc.admin1);
  document.getElementById("locationName").textContent = nameParts.join(", ");

  const cur = data.current;
  document.getElementById("currentTemp").textContent = Math.round(cur.temperature_C) + "°";
  document.getElementById("condition").textContent = cur.condition;
  document.getElementById("feelsLike").textContent =
    cur.feels_like_C != null ? `Feels like ${Math.round(cur.feels_like_C)}°` : "";

  const today = data.daily_outlook[0];
  if (today) {
    document.getElementById("hilo").textContent =
      `H:${Math.round(today.temp_max_C)}°  L:${Math.round(today.temp_min_C)}°`;
  }

  // Details card
  document.getElementById("humidity").textContent = Math.round(cur.humidity_pct) + "%";
  document.getElementById("dewPoint").textContent =
    cur.dew_point_C != null ? Math.round(cur.dew_point_C) + "°" : "—";
  document.getElementById("pressure").textContent =
    cur.pressure_hpa != null ? Math.round(cur.pressure_hpa) + " hPa" : "—";
  document.getElementById("visibility").textContent =
    cur.visibility_km != null ? cur.visibility_km.toFixed(1) + " km" : "—";

  // Wind
  document.getElementById("windSpeed").textContent = Math.round(cur.wind_speed_kmh) + " km/h";

  // Precipitation card
  document.getElementById("precipToday").textContent = (today ? today.rain_mm.toFixed(1) : "0.0") + " mm";
  const nextRainDay = data.daily_outlook.find((d) => d.rain_mm > 0);
  document.getElementById("precipNext").textContent = nextRainDay
    ? `Next rain expected ${fmtDay(nextRainDay.date, data.daily_outlook.indexOf(nextRainDay))}: ${nextRainDay.rain_mm.toFixed(1)} mm`
    : "No rain expected in the next week";

  // UV
  const uv = uvCategory(cur.uv_index);
  document.getElementById("uvValue").textContent = cur.uv_index != null ? cur.uv_index : "—";
  document.getElementById("uvCategory").textContent = uv.text;
  const uvBar = document.getElementById("uvBar");
  uvBar.style.width = Math.min(uv.pct, 100) + "%";
  uvBar.style.background = uv.color;

  // Air quality
  const aqiCard = document.getElementById("aqiBox");
  if (data.air_quality) {
    const aqi = data.air_quality;
    document.getElementById("aqiValue").textContent = Math.round(aqi.us_aqi);
    document.getElementById("aqiCategory").textContent = aqi.category;
    const aqiBar = document.getElementById("aqiBar");
    aqiBar.style.width = Math.min((aqi.us_aqi / 300) * 100, 100) + "%";
    aqiBar.style.background = aqiColor(aqi.category);
    aqiCard.hidden = false;
  } else {
    document.getElementById("aqiValue").textContent = "—";
    document.getElementById("aqiCategory").textContent = "Unavailable for this location";
    document.getElementById("aqiBar").style.width = "0%";
  }

  // Sun & moon
  document.getElementById("sunrise").textContent = fmtClock(data.sun.sunrise);
  document.getElementById("sunset").textContent = fmtClock(data.sun.sunset);
  document.getElementById("moonPhase").textContent = data.moon.name;
  document.getElementById("moonIllum").textContent = data.moon.illumination_pct + "%";

  // Hourly strip
  const hourlyStrip = document.getElementById("hourlyStrip");
  hourlyStrip.innerHTML = "";
  data.hourly_outlook.forEach((h, i) => {
    const el = document.createElement("div");
    el.className = "hour-item";
    el.innerHTML = `
      <div class="h-time">${i === 0 ? "Now" : fmtHour(h.time)}</div>
      <div class="h-icon">${iconFor(h.condition)}</div>
      <div class="h-temp">${Math.round(h.temp_C)}°</div>
    `;
    hourlyStrip.appendChild(el);
  });

  // Daily list
  const dailyList = document.getElementById("dailyList");
  dailyList.innerHTML = "";
  data.daily_outlook.forEach((d, i) => {
    const row = document.createElement("div");
    row.className = "day-row";
    row.innerHTML = `
      <div class="d-day">${fmtDay(d.date, i)}</div>
      <div class="d-icon">${iconFor(d.condition)}</div>
      <div class="d-rain">${d.rain_mm > 0 ? d.rain_mm.toFixed(1) + " mm" : ""}</div>
      <div class="d-range">${Math.round(d.temp_min_C)}° - ${Math.round(d.temp_max_C)}°</div>
    `;
    dailyList.appendChild(row);
  });

  // Flood banner
  const floodBanner = document.getElementById("floodBanner");
  if (data.flood_risk_available && data.flood_risk) {
    floodBanner.hidden = false;
    const fr = data.flood_risk;
    const pct = Math.round(fr.flood_risk_probability * 100);

    floodBanner.className = "flood-banner band-" + fr.risk_band;

    const gauge = document.getElementById("floodGauge");
    gauge.style.setProperty("--gauge-pct", pct);
    gauge.style.setProperty("--gauge-color", talukColor(fr.risk_band));

    document.getElementById("floodPct").textContent = pct + "%";
    const bandEl = document.getElementById("floodBand");
    bandEl.textContent = fr.risk_band;
    bandEl.className = "gauge-band " + fr.risk_band;

    const dateLabel = new Date(fr.date + "T00:00:00").toLocaleDateString(undefined, {
      weekday: "short", month: "short", day: "numeric",
    });
    const sourceNote = fr.bridge_days > 0 ? (fr.is_live_input ? " · live forecast" : " · trend-based") : "";
    document.getElementById("floodSub").textContent = dateLabel + sourceNote;
    document.getElementById("floodTalukTag").textContent = fr.taluk ? fr.taluk + " taluk" : "";

    initFloodTalukMap();
    initOutlookChart();
    initClimatologyChart();
  } else {
    floodBanner.hidden = true;
  }

  // Map
  initMap(loc.lat, loc.lon, nameParts.join(", "));
}

// ---------- Fetching ----------

async function loadByCoords(lat, lon, name, admin1, country) {
  const requestId = ++currentRequestId;
  setStatus(`Loading weather for ${name || "your location"}…`);
  try {
    const params = new URLSearchParams({ lat, lon });
    if (name) params.set("name", name);
    if (admin1) params.set("admin1", admin1);
    if (country) params.set("country", country);
    const resp = await fetch(`/weather-by-coords?${params.toString()}`);
    if (!resp.ok) throw new Error((await resp.json()).detail || "Request failed");
    const json = await resp.json();
    if (requestId !== currentRequestId) return; // a newer request has since started - discard this stale one
    render(json);
  } catch (err) {
    if (requestId !== currentRequestId) return;
    setStatus(`Couldn't load weather: ${err.message}`, true);
  }
}

async function loadByName(query) {
  const requestId = ++currentRequestId;
  setStatus(`Looking up "${query}"…`);
  try {
    const resp = await fetch(`/weather/${encodeURIComponent(query)}`);
    if (!resp.ok) throw new Error((await resp.json()).detail || "Request failed");
    const json = await resp.json();
    if (requestId !== currentRequestId) return;
    render(json);
  } catch (err) {
    if (requestId !== currentRequestId) return;
    setStatus(`Couldn't load weather: ${err.message}`, true);
  }
}

// ---------- Search autocomplete ----------

let debounceTimer = null;

function hideSuggestions() {
  suggestionsEl.hidden = true;
  suggestionsEl.innerHTML = "";
  activeSuggestionIndex = -1;
  currentSuggestions = [];
}

function renderSuggestions(results) {
  currentSuggestions = results;
  activeSuggestionIndex = -1;
  if (!results.length) {
    suggestionsEl.innerHTML = '<li class="no-results">No matches</li>';
    suggestionsEl.hidden = false;
    return;
  }
  suggestionsEl.innerHTML = results
    .map((r, i) => {
      const meta = [r.admin1, r.country].filter(Boolean).join(", ");
      return `<li data-index="${i}"><span class="s-name">${r.name}</span><span class="s-meta">${meta}</span></li>`;
    })
    .join("");
  suggestionsEl.hidden = false;
}

searchInput.addEventListener("input", () => {
  const q = searchInput.value.trim();
  clearTimeout(debounceTimer);
  if (q.length < 2) {
    hideSuggestions();
    return;
  }
  debounceTimer = setTimeout(async () => {
    try {
      const resp = await fetch(`/geocode-search?q=${encodeURIComponent(q)}`);
      const data = await resp.json();
      renderSuggestions(data.results || []);
    } catch (err) {
      hideSuggestions();
    }
  }, 300);
});

suggestionsEl.addEventListener("click", (e) => {
  const li = e.target.closest("li[data-index]");
  if (!li) return;
  const r = currentSuggestions[Number(li.dataset.index)];
  searchInput.value = r.name;
  hideSuggestions();
  loadByCoords(r.lat, r.lon, r.name, r.admin1, r.country);
});

searchInput.addEventListener("keydown", (e) => {
  const items = suggestionsEl.querySelectorAll("li[data-index]");
  if (!items.length) return;
  if (e.key === "ArrowDown") {
    e.preventDefault();
    activeSuggestionIndex = Math.min(activeSuggestionIndex + 1, items.length - 1);
  } else if (e.key === "ArrowUp") {
    e.preventDefault();
    activeSuggestionIndex = Math.max(activeSuggestionIndex - 1, 0);
  } else if (e.key === "Enter" && activeSuggestionIndex >= 0) {
    e.preventDefault();
    items[activeSuggestionIndex].dispatchEvent(new MouseEvent("click", { bubbles: true }));
    return;
  } else {
    return;
  }
  items.forEach((it, i) => it.classList.toggle("active", i === activeSuggestionIndex));
});

document.addEventListener("click", (e) => {
  if (!e.target.closest(".search-wrap")) hideSuggestions();
});

searchForm.addEventListener("submit", (e) => {
  e.preventDefault();
  const query = searchInput.value.trim();
  if (query) {
    hideSuggestions();
    loadByName(query);
  }
});

function useMyLocation() {
  if (!navigator.geolocation) {
    setStatus("Your browser doesn't support location detection. Search for a city instead.", true);
    return;
  }
  setStatus("Getting your location…");
  navigator.geolocation.getCurrentPosition(
    (pos) => loadByCoords(pos.coords.latitude, pos.coords.longitude, "Current Location", "", ""),
    () => loadByName("Kodagu"),
    { timeout: 8000 }
  );
}

locateBtn.addEventListener("click", useMyLocation);

// First load: try real GPS; if denied/unavailable, fall back to a default
// search so the page is never blank.
useMyLocation();
