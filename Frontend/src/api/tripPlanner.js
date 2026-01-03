export const createTripPlan = async (places) => {
    const response = await fetch("http://localhost:3000/api/trip-planner", {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify({ places })
    });
  
    if (!response.ok) {
      throw new Error("Failed to create trip plan");
    }
  
    return response.json();
  };
  