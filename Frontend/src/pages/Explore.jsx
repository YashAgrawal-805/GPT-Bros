import React, { useEffect, useRef, useState } from "react";
import { useSelector, useDispatch } from "react-redux";
import { initSocket, getSocket } from "../controllers/UseSocket";
import SoS from "../controllers/SoS";
import {
  addAcceptedRequest,
  addConnection,
  addSendRequest,
  removeSendRequest,
} from "../store/store";
import ExpNavbar from "../components/Explore/expNavbar";
import Footer from "../components/Footer";
import HeroAnimation from "../components/Explore/HeroAnimation";
import CardsParallaxAnimation from "../components/Explore/parallex";
import AlertWatcher from "../utility/AlertWatcher";
import Notification from "../utility/Notifications"; // ✅ Import

const Explore = ({ theme, setTheme }) => {
  const dispatch = useDispatch();

  // Redux state
  const { lat, lng } = useSelector((state) => state.app.latLng);
  const { isTrack, sendRequests } = useSelector((state) => state.app);

  // 🔹 Notification State
  const [notification, setNotification] = useState("");

  const showNotification = (msg) => {
    setNotification(msg);
  };

  const closeNotification = () => {
    setNotification("");
  };

  // ref to keep latest sendRequests
  const sendRequestsRef = useRef(sendRequests);

  useEffect(() => {
    sendRequestsRef.current = sendRequests;
  }, [sendRequests]);

  // 🔹 Emit location periodically
  useEffect(() => {
    const token = localStorage.getItem("token");
    initSocket(token);
    const socket = getSocket();
    if (!lat || !lng || !socket || !isTrack) return;

    console.log("Sending live location:", lat, lng);
    socket.emit("location", { latitude: lat, longitude: lng });

    const interval = setInterval(() => {
      socket.emit("location", { latitude: lat, longitude: lng });
      console.log("Re-sent location:", lat, lng);
    }, 10000);

    return () => clearInterval(interval);
  }, [lat, lng, isTrack]);

  useEffect(() => {
    const handleTwitterAlert = (e) => {
      const data = e.detail;
      showNotification(`🚨 Twitter Alert Detected`);
    };
  
    window.addEventListener("TWITTER_ALERT", handleTwitterAlert);
  
    return () => {
      window.removeEventListener("TWITTER_ALERT", handleTwitterAlert);
    };
  }, []);
  

  // 🔹 Handle socket events
  useEffect(() => {
    const socket = getSocket();
    if (!socket) return;

    const handleSoloTraveler = (data) => {
      const existing = sendRequestsRef.current.find(
        (traveler) => traveler.name === data.username
      );

      if (existing) {
        dispatch(removeSendRequest(existing.id));
        console.log(`Removed solo traveler: ${existing.name}`);
      } else {
        console.log("New solo traveler nearby:", data);
        dispatch(
          addSendRequest({
            id: data.id,
            username: data.username,
            distance: data.distance,
          })
        );
        showNotification(`New solo traveler nearby: ${data.username}`);
      }
    };

    const handleSoloRequestReceived = (data) => {
      console.log("Solo request received:", data);
      dispatch(
        addAcceptedRequest({
          id: data.from,
          name: data.name,
          distance: data.distance || null,
        })
      );
      dispatch(removeSendRequest(data.from));
      showNotification(`Request Received from ${data.name}`);
    };

    const handleSosAlert = (data) => {
      const { from, location } = data;
      const locString = location
        ? `(${location.latitude.toFixed(5)}, ${location.longitude.toFixed(5)})`
        : "unknown location";

      showNotification(`🚨 SOS Alert from ${from} at location: ${locString}`);
    };

    const handleSoloRequestAccepted = (data) => {
      dispatch(
        addConnection({
          id: data.by,
          name: data.username,
        })
      );
      showNotification(`✅ Request Accepted by ${data.username}`);
    };

    const handleNearbyNotification = (data) => {
      showNotification(data.message);
    };

    const handleDistanceNotifications = (notifications) => {
      if (Array.isArray(notifications) && notifications.length > 0) {
        notifications.forEach((notification) => {
          const { message } = notification;
          if (message) showNotification(message);
        });
      }
    };

    socket.on("soloTravelerNearby", handleSoloTraveler);
    socket.on("soloRequestReceived", handleSoloRequestReceived);
    socket.on("sosAlert", handleSosAlert);
    socket.on("soloRequestAccepted", handleSoloRequestAccepted);
    socket.on("nearbyNotification", handleNearbyNotification);
    socket.on("distanceNotifications", handleDistanceNotifications);

    return () => {
      socket.off("soloTravelerNearby", handleSoloTraveler);
      socket.off("soloRequestReceived", handleSoloRequestReceived);
      socket.off("sosAlert", handleSosAlert);
      socket.off("soloRequestAccepted", handleSoloRequestAccepted);
    };
  }, [dispatch]);

  const gridColor =
    theme === "dark" ? "rgba(255,255,255,0.05)" : "rgba(0,0,0,0.05)";

  return (
    <div
      className="relative min-h-screen"
      style={{
        backgroundColor: theme === "dark" ? "#000000" : "#ffffff",
        backgroundImage: `
          linear-gradient(to right, ${gridColor} 1px, transparent 1px),
          linear-gradient(to bottom, ${gridColor} 1px, transparent 1px)
        `,
        backgroundSize: "80px 80px",
      }}
    >
      <SoS />
      <AlertWatcher/>
      <ExpNavbar theme={theme} setTheme={setTheme} />
      <HeroAnimation theme={theme} />
      <CardsParallaxAnimation theme={theme} />
      <Footer />

      {/* ✅ Beautiful Notification */}
      <Notification
        message={notification}
        onClose={closeNotification}
        duration={5000}
      />
    </div>
  );
};

export default Explore;
