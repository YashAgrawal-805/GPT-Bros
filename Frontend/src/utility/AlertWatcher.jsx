import { useEffect, useRef } from "react";
import { useSelector } from "react-redux";
import { twitterData } from "../api/twitterData";

const TEN_MINUTES = 10 * 60 * 1000;

const AlertWatcher = () => {
  const isAlert = useSelector((state) => state.app.isAlert);
  const { lat, lng } = useSelector((state) => state.app.latLng);

  const intervalRef = useRef(null);

  useEffect(() => {
    // stop polling if alert OFF or location missing
    if (!isAlert || lat == null || lng == null) {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
        intervalRef.current = null;
      }
      return;
    }

    const fetchAlerts = async () => {
      try {
        const result = await twitterData({ lat, lng });

        if (result && result.length > 0) {
          // 🔔 trigger popup / notification
          window.dispatchEvent(
            new CustomEvent("TWITTER_ALERT", { detail: result })
          );
        }
      } catch (err) {
        console.error("Twitter alert error:", err);
      }
    };

    // run immediately
    fetchAlerts();

    // run every 10 min
    intervalRef.current = setInterval(fetchAlerts, TEN_MINUTES);

    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
        intervalRef.current = null;
      }
    };
  }, [isAlert, lat, lng]);

  return null;
};

export default AlertWatcher;
