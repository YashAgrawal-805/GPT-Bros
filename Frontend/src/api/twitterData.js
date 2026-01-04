export const twitterData = async ({ lat, lng }) => {
    console.log("Mock twitterData called at:", lat, lng);
  
    return [
      {
        id: 1,
        message: "🚨 Protest reported near your location",
        severity: "high",
      },
    ];
  };
  