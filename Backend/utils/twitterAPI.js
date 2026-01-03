async function twitterData() {
    try {
      const res = await fetch("http://localhost:5000/get-tweets?hashtag=delhi");
      const data = await res.json();
      console.log(data);
      return data;
    } catch (err) {
      console.error("Error fetching tweets:", err);
      return null;
    }
  }

  module.exports = { twitterData }; 
  

