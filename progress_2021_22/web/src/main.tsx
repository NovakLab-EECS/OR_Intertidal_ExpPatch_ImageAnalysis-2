import { Provider } from 'jotai';
import React from 'react';
import ReactDOM from 'react-dom/client';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { ToastContainer } from 'react-toastify';
import BoundingBoxes from './pages/BoundingBoxes';
import Draw from './pages/Draw';
import DrawBB from './pages/DrawBB';
import App from './pages/Home';
import Verify from './pages/Verify';
import './styles/index.css';

import 'react-toastify/dist/ReactToastify.css';

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <Provider>
      <Router>
        <Routes>
          <Route path="/" element={<App />} />
          <Route path="/inferences" element={<Verify />} />
          <Route path="/boundingBoxes" element={<BoundingBoxes />} />
          <Route path="/bboxes/:folder" element={<Draw />} />
          <Route path="/draw/:folder/:image" element={<DrawBB />} />
        </Routes>
      </Router>
      <ToastContainer />
    </Provider>
  </React.StrictMode>
);
