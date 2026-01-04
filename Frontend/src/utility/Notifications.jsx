// components/Notification.js
import React, { useEffect, useState } from "react";
import { motion, AnimatePresence } from "framer-motion";

const Notification = ({ message, onClose, duration = 5000 }) => {
  const [progress, setProgress] = useState(100);

  useEffect(() => {
    if (!message) return;

    // Timer for auto-close
    const timer = setTimeout(() => {
      onClose();
    }, duration);

    // Progress bar countdown
    const interval = setInterval(() => {
      setProgress((prev) => (prev > 0 ? prev - 100 / (duration / 100) : 0));
    }, 100);

    return () => {
      clearTimeout(timer);
      clearInterval(interval);
    };
  }, [message, duration, onClose]);

  return (
    <AnimatePresence>
      {message && (
        <motion.div
          initial={{ y: -100, opacity: 0 }}
          animate={{ y: 0, opacity: 1 }}
          exit={{ y: -100, opacity: 0 }}
          transition={{ type: "spring", stiffness: 100 }}
          className="fixed top-5 left-1/2 -translate-x-1/2 z-50 w-[90%] max-w-md bg-white dark:bg-gray-900 shadow-xl rounded-xl p-4 text-center"
        >
          <p className="text-gray-900 dark:text-gray-100 text-lg">{message}</p>
          <button
            onClick={onClose}
            className="mt-3 px-4 py-1 rounded-lg bg-blue-600 text-white hover:bg-blue-700 transition"
          >
            OK
          </button>

          {/* Progress bar */}
          <div className="mt-3 h-1 w-full bg-gray-300 rounded">
            <motion.div
              initial={{ width: "100%" }}
              animate={{ width: `${progress}%` }}
              transition={{ duration: 0.1 }}
              className="h-1 bg-blue-500 rounded"
            />
          </div>
        </motion.div>
      )}
    </AnimatePresence>
  );
};

export default Notification;
