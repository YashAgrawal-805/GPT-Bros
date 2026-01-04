export const createTripPlan = async ({
  lat,
  lng,
  date,
  radius_km,
  max_stops = 6,
  start_hour = 8,
  end_hour = 20,
  preferred_places,
  include_nearby = true
}) => {
  const response = await fetch("/plan-day", {
    method: "POST",
    headers: {
      "Accept": "application/json",
      "Content-Type": "application/json"
    },
    body: JSON.stringify({
      lat,
      lng,
      date,
      radius_km,
      max_stops,
      start_hour,
      end_hour,
      preferred_places,
      include_nearby
    })
  });
  console.log(lat, lng, date, radius_km, max_stops, start_hour, end_hour, preferred_places, include_nearby);
  console.log("Response status:", response.status);
  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(`Failed to create trip plan: ${errorText}`);
  }

  return response.json();
};
