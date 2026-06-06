const form = document.querySelector("#predictForm");
const prediction = document.querySelector("#prediction");
const category = document.querySelector("#category");
const progressRing = document.querySelector("#progressRing");
const recommendations = document.querySelector("#recommendations");
const fillIndonesia = document.querySelector("#fillIndonesia");
const circumference = 2 * Math.PI * 56;

progressRing.style.strokeDasharray = `${circumference}`;
progressRing.style.strokeDashoffset = `${circumference}`;

const indonesiaPreset = {
  Country: "Indonesia",
  Status: "Developing",
  Year: 2015,
  "Adult Mortality": 176,
  "infant deaths": 114,
  Alcohol: 0.08,
  "percentage expenditure": 0,
  "Hepatitis B": 78,
  "Measles ": 15099,
  " BMI ": 27.1,
  "under-five deaths ": 136,
  Polio: 78,
  "Total expenditure": 2.87,
  "Diphtheria ": 78,
  " HIV/AIDS": 0.3,
  GDP: 861.4,
  Population: 258162113,
  " thinness  1-19 years": 1.4,
  " thinness 5-9 years": 1.2,
  "Income composition of resources": 0.686,
  Schooling: 12.9,
};

function formDataToObject() {
  const data = new FormData(form);
  const payload = {};
  for (const [key, value] of data.entries()) {
    payload[key] = Number.isNaN(Number(value)) || value.trim() === "" ? value : Number(value);
  }
  return payload;
}

function setMeter(value) {
  const ratio = Math.max(0, Math.min(1, (value - 35) / 60));
  progressRing.style.strokeDashoffset = `${circumference * (1 - ratio)}`;
}

function renderRecommendations(items) {
  recommendations.innerHTML = "";
  if (!items.length) {
    const li = document.createElement("li");
    li.textContent = "Indikator utama sudah berada di atas median data latih.";
    recommendations.appendChild(li);
    return;
  }

  items.forEach((item) => {
    const li = document.createElement("li");
    li.innerHTML = `<strong>${item.title}</strong><span>${item.detail}</span>`;
    recommendations.appendChild(li);
  });
}

fillIndonesia.addEventListener("click", () => {
  Object.entries(indonesiaPreset).forEach(([key, value]) => {
    const field = form.elements[key];
    if (field) field.value = value;
  });
});

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  category.textContent = "Memproses";
  category.classList.add("loading");

  const response = await fetch("/predict", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(formDataToObject()),
  });
  const result = await response.json();

  prediction.textContent = result.life_expectancy.toFixed(2);
  category.textContent = result.category;
  category.classList.remove("loading");
  setMeter(result.life_expectancy);
  renderRecommendations(result.recommendations);
});
