// src/components/interfaces/PlaceSelectionInterface.jsx
import React, { useState } from 'react';
import { motion } from 'framer-motion';

const PlaceCard = ({ place, theme, onAdd, isAdded, isMobile }) => (
  <motion.div
    whileHover={{ scale: 1.02, y: -2 }}
    whileTap={{ scale: 0.98 }}
    style={{
      width: '100%',
      maxWidth: isMobile ? '250px' : '220px',
      minWidth: isMobile ? '140px' : '160px',
      height: isMobile ? '220px' : '260px',
      borderRadius: '16px',
      background: theme === 'dark'
        ? 'linear-gradient(135deg, #1e293b 0%, #334155 50%, #475569 100%)'
        : 'linear-gradient(135deg, #ffffff 0%, #f8fafc 50%, #f1f5f9 100%)',
      border: theme === 'dark' ? '1px solid #475569' : '1px solid #e2e8f0',
      boxShadow: theme === 'dark'
        ? '0 8px 25px rgba(0,0,0,0.3)'
        : '0 8px 25px rgba(0,0,0,0.1)',
      padding: '14px',
      display: 'flex',
      flexDirection: 'column',
      cursor: 'pointer',
      transition: 'all 0.3s ease',
      opacity: isAdded ? 0.6 : 1
    }}
  >
    <div style={{
      width: '100%',
      height: isMobile ? '100px' : '140px',
      borderRadius: '12px',
      overflow: 'hidden',
      marginBottom: '10px'
    }}>
      <img
        src={place.image}
        alt={place.name}
        style={{
          width: '100%',
          height: '100%',
          objectFit: 'cover'
        }}
      />
    </div>
    <h3 style={{
      fontSize: isMobile ? '14px' : '16px',
      fontWeight: '600',
      color: theme === 'dark' ? '#ffffff' : '#374151',
      margin: '0 0 10px 0',
      textAlign: 'center',
      flexGrow: 1,
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center'
    }}>
      {place.name}
    </h3>
    <motion.button
      whileHover={!isAdded ? { scale: 1.05 } : {}}
      whileTap={!isAdded ? { scale: 0.95 } : {}}
      onClick={() => !isAdded && onAdd(place)}
      disabled={isAdded}
      style={{
        padding: '8px 16px',
        borderRadius: '20px',
        background: isAdded ? '#10b981' : '#6366f1',
        color: '#fff',
        border: 'none',
        cursor: isAdded ? 'not-allowed' : 'pointer',
        fontSize: '14px',
        fontWeight: '600',
        boxShadow: isAdded
          ? '0 4px 15px rgba(16, 185, 129, 0.3)'
          : '0 4px 15px rgba(99, 102, 241, 0.3)',
        transition: 'all 0.3s ease'
      }}
    >
      {isAdded ? 'ADDED' : 'ADD'}
    </motion.button>
  </motion.div>
);

const PlaceSelectionInterface = ({ theme, onBack, onCreatePlan, isMobile }) => {
  const [selectedPlaces, setSelectedPlaces] = useState([]);

const places = [
  {
    id: 1,
    name: "Rourkela Overview",
    image: "https://images.unsplash.com/photo-1506744038136-46273834b3fb?w=400&h=300&fit=crop" // generic city image
  },
  {
    id: 2,
    name: "Hanuman Vatika",
    image: "https://images.unsplash.com/photo-1568605114967-8130f3a36994?w=400&h=300&fit=crop" // temple/statue
  },
  {
    id: 3,
    name: "Vedvyas Temple",
    image: "https://images.unsplash.com/photo-1526483360610-c3f64c3b65ab?w=400&h=300&fit=crop" // temple/nature
  },
  {
    id: 4,
    name: "Vaishno Devi Temple (hilltop)",
    image: "https://images.unsplash.com/photo-1549887536-5c0a5f7482f2?w=400&h=300&fit=crop" // hilltop temple
  },
  {
    id: 5,
    name: "Laxmi Narayan Mandir",
    image: "https://images.unsplash.com/photo-1508997449629-303059a0394b?w=400&h=300&fit=crop" // temple
  },
  {
    id: 6,
    name: "Rani Sati Temple",
    image: "https://images.unsplash.com/photo-1505672678657-cc7037095e11?w=400&h=300&fit=crop" // temple
  },
  {
    id: 7,
    name: "Vaishno Devi Temple (Sector 5)",
    image: "https://images.unsplash.com/photo-1506744038136-46273834b3fb?w=400&h=300&fit=crop" // temple, reused generic image
  },
  {
    id: 8,
    name: "Jagannath Temple, Sector 3",
    image: "https://images.unsplash.com/photo-1523301343968-2192e6a7d08b?w=400&h=300&fit=crop" // temple
  },
  {
    id: 9,
    name: "Bhairab Mandir",
    image: "https://images.unsplash.com/photo-1506748686214-e9df14d4d9d0?w=400&h=300&fit=crop" // temple nature
  },
  {
    id: 10,
    name: "Indira Gandhi Park",
    image: "https://images.unsplash.com/photo-1500534623283-312aade485b7?w=400&h=300&fit=crop" // park
  },
  {
    id: 11,
    name: "Ispat Nehru Park",
    image: "https://images.unsplash.com/photo-1470770903676-69b98201ea1c?w=400&h=300&fit=crop" // park
  },
  {
    id: 12,
    name: "Mandira Dam",
    image: "https://images.unsplash.com/photo-1506744038136-46273834b3fb?w=400&h=300&fit=crop" // dam, reused generic city image or you can change
  },
  {
    id: 13,
    name: "Pitamahal Dam",
    image: "https://images.unsplash.com/photo-1494526585095-c41746248156?w=400&h=300&fit=crop" // water reservoir
  },
  {
    id: 14,
    name: "Darjeeng Picnic Spot",
    image: "https://images.unsplash.com/photo-1506744038136-46273834b3fb?w=400&h=300&fit=crop" // nature/picnic generic
  },
  {
    id: 15,
    name: "Ghogar Natural Site",
    image: "https://images.unsplash.com/photo-1506744038136-46273834b3fb?w=400&h=300&fit=crop" // natural site
  },
  {
    id: 16,
    name: "Kanha Kund",
    image: "https://images.unsplash.com/photo-1506744038136-46273834b3fb?w=400&h=300&fit=crop" // water/forest
  },
  {
    id: 17,
    name: "Ghagara Waterfall",
    image: "https://images.unsplash.com/photo-1506744038136-46273834b3fb?w=400&h=300&fit=crop" // waterfall
  },
  {
    id: 18,
    name: "Ushakothi Wildlife Sanctuary",
    image: "https://images.unsplash.com/photo-1506744038136-46273834b3fb?w=400&h=300&fit=crop" // wildlife park generic
  },
  {
    id: 19,
    name: "Green Park",
    image: "https://images.unsplash.com/photo-1506744038136-46273834b3fb?w=400&h=300&fit=crop" // park
  },
  {
    id: 20,
    name: "Koel Riverbank",
    image: "https://images.unsplash.com/photo-1506744038136-46273834b3fb?w=400&h=300&fit=crop" // riverbank
  },
  {
    id: 21,
    name: "Rani Sati Mandir",
    image: "https://images.unsplash.com/photo-1505672678657-cc7037095e11?w=400&h=300&fit=crop" // temple reused
  },
  {
    id: 22,
    name: "Deodhar Picnic Spot",
    image: "https://images.unsplash.com/photo-1506744038136-46273834b3fb?w=400&h=300&fit=crop" // picnic spot
  },
  {
    id: 23,
    name: "Mirigikhoj Waterfall",
    image: "https://images.unsplash.com/photo-1506744038136-46273834b3fb?w=400&h=300&fit=crop" // waterfall
  }
];

  const handleAddPlace = (place) => {
    if (!selectedPlaces.find(p => p.id === place.id)) {
      setSelectedPlaces([...selectedPlaces, place]);
    }
  };

  const handleCreatePlan = () => {
    onCreatePlan(selectedPlaces);
  };

  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.95 }}
      animate={{ opacity: 1, scale: 1 }}
      exit={{ opacity: 0, scale: 0.95 }}
      transition={{ duration: 0.3 }}
      style={{
        position: 'absolute',
        top: 0,
        left: 0,
        width: '100%',
        height: '100%',
        background: theme === 'dark' ? 'linear-gradient(135deg, #0f172a 0%, #1e293b 25%, #334155 50%, #475569 75%, #64748b 100%)' : 'linear-gradient(135deg, #ffffff 0%, #f8fafc 25%, #f1f5f9 50%, #e2e8f0 75%, #cbd5e1 100%)',
        borderRadius: '25px',
        padding: isMobile ? '16px' : '24px',
        display: 'flex',
        flexDirection: 'column',
        zIndex: 20
      }}
    >
      <motion.button
        whileHover={{ scale: 1.1 }}
        whileTap={{ scale: 0.9 }}
        onClick={onBack}
        style={{
          position: 'absolute',
          top: isMobile ? '10px' : '20px',
          left: isMobile ? '10px' : '20px',
          width: '40px',
          height: '40px',
          borderRadius: '50%',
          background: theme === 'dark' ? '#374151' : '#f3f4f6',
          border: 'none',
          cursor: 'pointer',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          fontSize: '18px',
          color: theme === 'dark' ? '#ffffff' : '#374151',
          boxShadow: '0 4px 15px rgba(0,0,0,0.1)',
          zIndex: 30
        }}
      >
        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" style={{ color: theme === 'dark' ? '#ffffff' : '#000000' }}>
          <path d="M19 12H5M12 19l-7-7 7-7" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
        </svg>
      </motion.button>
      <motion.button
        whileHover={selectedPlaces.length > 0 ? { scale: 1.05, y: -2 } : {}}
        whileTap={selectedPlaces.length > 0 ? { scale: 0.95 } : {}}
        onClick={handleCreatePlan}
        disabled={selectedPlaces.length === 0}
        style={{
          position: 'absolute',
          bottom: isMobile ? '10px' : '20px',
          right: isMobile ? '10px' : '20px',
          padding: '12px 24px',
          borderRadius: '25px',
          background: selectedPlaces.length > 0 ? '#6366f1' : '#94a3b8',
          color: '#fff',
          border: 'none',
          cursor: selectedPlaces.length > 0 ? 'pointer' : 'not-allowed',
          fontSize: '14px',
          fontWeight: '600',
          boxShadow: selectedPlaces.length > 0 ? '0 4px 15px rgba(99, 102, 241, 0.3)' : '0 4px 15px rgba(148, 163, 184, 0.3)',
          zIndex: 30,
          transition: 'all 0.3s ease'
        }}
      >
        Create Plan {selectedPlaces.length > 0 && `(${selectedPlaces.length})`}
      </motion.button>
      <div style={{ textAlign: 'center', marginBottom: isMobile ? '20px' : '30px', marginTop: '60px' }}>
        <h2 style={{ fontSize: isMobile ? '24px' : '32px', fontWeight: '700', color: theme === 'dark' ? '#ffffff' : '#374151', margin: '0' }}>
          Choose places you want to explore!
        </h2>
      </div>
      <div
        style={{
          flex: 1,
          overflowY: 'auto',
          overflowX: 'hidden',
          paddingBottom: '100px',
          paddingTop: '10px',
          scrollbarWidth: 'none',
          msOverflowStyle: 'none'
        }}
        className="hide-scrollbar"
      >
        <style jsx>{`
          .hide-scrollbar::-webkit-scrollbar {
            display: none;
          }
        `}</style>
        <div style={{
          display: 'grid',
          gridTemplateColumns: isMobile ? 'repeat(auto-fit, minmax(140px, 1fr))' : 'repeat(auto-fit, minmax(200px, 1fr))',
          gap: isMobile ? '12px' : '20px',
          padding: '0 16px',
          justifyItems: 'center'
        }}>
          {places.map((place) => (
            <PlaceCard
              key={place.id}
              place={place}
              theme={theme}
              onAdd={handleAddPlace}
              isAdded={selectedPlaces.find(p => p.id === place.id)}
              isMobile={isMobile}
            />
          ))}
        </div>
      </div>
    </motion.div>
  );
};

export default PlaceSelectionInterface;