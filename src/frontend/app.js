// Vanilla JS, no build step, no external libraries - talks to the FastAPI
// backend served from the same origin as this page, so these are always
// relative URLs (works identically on localhost and after deployment).

const statusBox = document.getElementById("statusBox");
const weatherCard = document.getElementById("weatherCard");
const homeLabel = document.getElementById("homeLabel");
const searchForm = document.getElementById("searchForm");
const searchInput = document.getElementById("searchInput");

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
  weatherCard.hidden = true;
  statusBox.textContent = text;
  statusBox.className = "status-box" + (isError ? " error-text" : "");
}

function fmtHour(iso) {
  const h = iso.slice(11, 16);
  return h;
}

function fmtDay(iso, index) {
  if (index === 0) return "Today";
  const d = new Date(iso + "T00:00:00");
  return d.toLocaleDateString(undefined, { weekday: "short" });
}

function render(data) {
  statusBox.hidden = true;
  weatherCard.hidden = false;

  const loc = data.location;
  const nameParts = [loc.name];
  if (loc.admin1) nameParts.push(loc.admin1);
  document.getElementById("locationName").textContent = nameParts.join(", ");

  document.getElementById("currentTemp").textContent =
    Math.round(data.current.temperature_C) + "°";
  document.getElementById("condition").textContent = data.current.condition;

  const today = data.daily_outlook[0];
  if (today) {
    document.getElementById("hilo").textContent =
      `H:${Math.round(today.temp_max_C)}°  L:${Math.round(today.temp_min_C)}°`;
  }

  document.getElementById("humidity").textContent = Math.round(data.current.humidity_pct) + "%";
  document.getElementById("wind").textContent = Math.round(data.current.wind_speed_kmh) + " km/h";

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

  const floodCard = document.getElementById("floodCard");
  if (data.flood_risk_available && data.flood_risk) {
    floodCard.hidden = false;
    const fr = data.flood_risk;
    const bandEl = document.getElementById("floodBand");
    bandEl.textContent = `${fr.risk_band} (${Math.round(fr.flood_risk_probability * 100)}%)`;
    bandEl.className = "flood-band " + fr.risk_band;
    document.getElementById("floodSub").textContent =
      `Most recent real data: ${fr.most_recent_real_data_date}`;
    const why = document.getElementById("floodWhy");
    why.innerHTML = fr.explanation
      .map(e => `<div><span class="${e.direction}">${e.direction === "up" ? "↑" : "↓"} ${e.feature}</span></div>`)
      .join("");
  } else {
    floodCard.hidden = true;
  }
}

async function loadByCoords(lat, lon) {
  homeLabel.hidden = false;
  setStatus("Loading weather for your location…");
  try {
    const resp = await fetch(`/weather-by-coords?lat=${lat}&lon=${lon}`);
    if (!resp.ok) throw new Error((await resp.json()).detail || "Request failed");
    render(await resp.json());
  } catch (err) {
    setStatus(`Couldn't load weather: ${err.message}`, true);
  }
}

async function loadByName(query) {
  homeLabel.hidden = true;
  setStatus(`Looking up "${query}"…`);
  try {
    const resp = await fetch(`/weather/${encodeURIComponent(query)}`);
    if (!resp.ok) throw new Error((await resp.json()).detail || "Request failed");
    render(await resp.json());
  } catch (err) {
    setStatus(`Couldn't load weather: ${err.message}`, true);
  }
}

searchForm.addEventListener("submit", (e) => {
  e.preventDefault();
  const query = searchInput.value.trim();
  if (query) loadByName(query);
});

// On first load: try the browser's real GPS location (like a phone weather
// app's "Home" screen). If denied or unavailable, fall back to a default
// search so the page is never blank.
if (navigator.geolocation) {
  navigator.geolocation.getCurrentPosition(
    (pos) => loadByCoords(pos.coords.latitude, pos.coords.longitude),
    () => loadByName("Kodagu"),
    { timeout: 8000 }
  );
} else {
  loadByName("Kodagu");
}
