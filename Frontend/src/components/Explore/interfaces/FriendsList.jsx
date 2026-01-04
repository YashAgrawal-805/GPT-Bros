import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { motion } from 'framer-motion';
import { io } from 'socket.io-client';

const socket = io("http://localhost:3000", {
  withCredentials: true,
});

const FriendCard = ({ person, theme }) => (
  <motion.div
    whileHover={{ scale: 1.02, y: -2 }}
    style={{
      background: theme === 'dark'
        ? 'linear-gradient(135deg, #1e293b 0%, #334155 100%)'
        : 'linear-gradient(135deg, #ffffff 0%, #f8fafc 100%)',
      border: theme === 'dark' ? '1px solid #475569' : '1px solid #e2e8f0',
      borderRadius: '15px',
      padding: '16px',
      marginBottom: '12px',
      minWidth: '0',
      boxShadow: theme === 'dark'
        ? '0 4px 15px rgba(0,0,0,0.3)'
        : '0 4px 15px rgba(0,0,0,0.1)'
    }}
  >
    <div style={{ display: 'flex', alignItems: 'center', gap: '16px' }}>
      <div style={{
        width: '50px',
        height: '50px',
        borderRadius: '50%',
        background: 'linear-gradient(135deg, #6366f1, #8b5cf6)',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        boxShadow: '0 4px 15px rgba(99, 102, 241, 0.3)'
      }}>
        <svg width="24" height="24" viewBox="0 0 24 24" fill="none" style={{ color: '#ffffff' }}>
          <path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
          <circle cx="12" cy="7" r="4" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
        </svg>
      </div>
      <div style={{ flex: 1, minWidth: '0' }}>
        <h4 style={{
          margin: '0 0 4px 0',
          fontSize: '16px',
          fontWeight: '600',
          color: theme === 'dark' ? '#ffffff' : '#000000',
          overflow: 'hidden',
          textOverflow: 'ellipsis',
          whiteSpace: 'nowrap'
        }}>
          {person.name}
        </h4>
        <p style={{
          margin: '0',
          fontSize: '14px',
          color: theme === 'dark' ? '#94a3b8' : '#6b7280',
          overflow: 'hidden',
          textOverflow: 'ellipsis',
          whiteSpace: 'nowrap'
        }}>
          {person.phone}
        </p>
      </div>
    </div>
  </motion.div>
);

const FriendSection = ({ title, data, theme, isMobile }) => (
  <div style={{
    flex: 1,
    minHeight: '0',
    display: 'flex',
    flexDirection: 'column'
  }}>
    <h3 style={{
      textAlign: 'center',
      margin: '0 0 20px 0',
      fontSize: isMobile ? '20px' : '24px',
      fontWeight: '700',
      color: theme === 'dark' ? '#ffffff' : '#000000',
    }}>
      {title}
    </h3>
    <div style={{
      flex: 1,
      overflowY: 'auto',
      paddingRight: '8px',
      scrollbarWidth: 'none',
      msOverflowStyle: 'none',
      maxHeight: isMobile ? '200px' : 'auto'
    }}>
      <style>
        {`
          div::-webkit-scrollbar {
            display: none;
          }
        `}
      </style>
      {data.map((person, index) => (
        <FriendCard
          key={index}
          person={person}
          theme={theme}
        />
      ))}
    </div>
  </div>
);

const FriendsList = ({ theme, isMobile }) => {
  const [connectionsList, setConnectionsList] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    const fetchConnections = async () => {
      try {
        const token = localStorage.getItem('token');
        const res = await axios.get('http://localhost:3000/api/my_accepted_connections', {
          headers: { Authorization: `Bearer ${token}` },
        });

        setConnectionsList(res.data.connections || []);
      } catch (err) {
        console.error('Error fetching connections:', err);
        setError('Failed to load connections');
      } finally {
        setLoading(false);
      }
    };

    fetchConnections();

    // 🔥 Socket listener for real-time updates
    socket.on("connectionsUpdated", (updatedConnections) => {
      console.log("Socket: connections updated", updatedConnections);
      setConnectionsList(updatedConnections);
    });

    return () => {
      socket.off("connectionsUpdated");
    };
  }, []);

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: 20 }}
      transition={{ duration: 0.5 }}
      style={{
        height: '100%',
        display: 'flex',
        flexDirection: 'column',
        marginTop: '15px',
        padding: '0px'
      }}
    >
      <h2 style={{
        textAlign: 'center',
        margin: '10px 0 10px 0',
        fontSize: isMobile ? '24px' : '32px',
        fontWeight: '800',
        color: theme === 'dark' ? '#ffffff' : '#000000'
      }}>
        <span style={{ color: '#6366f1' }}>Friend</span> List
      </h2>

      <div style={{
        flex: 1,
        display: 'flex',
        flexDirection: isMobile ? 'column' : 'row',
        gap: isMobile ? '20px' : '40px',
        minHeight: '0',
        paddingBottom: '20px'
      }}>
        {loading ? (
          <div style={{ textAlign: 'center', color: theme === 'dark' ? '#fff' : '#000' }}>
            Loading connections...
          </div>
        ) : error ? (
          <div style={{ textAlign: 'center', color: 'red' }}>{error}</div>
        ) : (
          <FriendSection
            title="Connections"
            data={connectionsList}
            theme={theme}
            isMobile={isMobile}
          />
        )}
      </div>
    </motion.div>
  );
};

export default FriendsList;