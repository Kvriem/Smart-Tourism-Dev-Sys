import React from 'react';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  BarElement,
  LineElement,
  PointElement,
  ArcElement,
  Title,
  Tooltip,
  Legend,
} from 'chart.js';
import { Bar, Line, Doughnut } from 'react-chartjs-2';
import './Dashboard.css';

ChartJS.register(
  CategoryScale,
  LinearScale,
  BarElement,
  LineElement,
  PointElement,
  ArcElement,
  Title,
  Tooltip,
  Legend
);

const Dashboard = ({ data }) => {
  const chartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        position: 'top',
        labels: {
          usePointStyle: true,
          padding: 20,
        },
      },
      tooltip: {
        backgroundColor: 'rgba(0, 0, 0, 0.8)',
        titleColor: '#fff',
        bodyColor: '#fff',
        borderColor: '#4f46e5',
        borderWidth: 1,
      },
    },
    scales: {
      x: {
        grid: {
          color: 'rgba(0, 0, 0, 0.1)',
        },
      },
      y: {
        grid: {
          color: 'rgba(0, 0, 0, 0.1)',
        },
      },
    },
  };

  const ratingData = {
    labels: ['1 Star', '2 Stars', '3 Stars', '4 Stars', '5 Stars'],
    datasets: [
      {
        label: 'Rating Distribution',
        data: data.ratingDistribution || [5, 10, 25, 35, 25],
        backgroundColor: [
          '#ef4444',
          '#f97316',
          '#eab308',
          '#22c55e',
          '#10b981',
        ],
        borderColor: [
          '#dc2626',
          '#ea580c',
          '#ca8a04',
          '#16a34a',
          '#059669',
        ],
        borderWidth: 2,
        borderRadius: 8,
      },
    ],
  };

  const monthlyData = {
    labels: ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun'],
    datasets: [
      {
        label: 'Average Rating',
        data: data.monthlyRatings || [4.2, 4.1, 4.3, 4.5, 4.4, 4.6],
        borderColor: '#4f46e5',
        backgroundColor: 'rgba(79, 70, 229, 0.1)',
        borderWidth: 3,
        fill: true,
        tension: 0.4,
        pointBackgroundColor: '#4f46e5',
        pointBorderColor: '#fff',
        pointBorderWidth: 2,
        pointRadius: 6,
      },
    ],
  };

  const departmentData = {
    labels: ['Front Desk', 'Housekeeping', 'Restaurant', 'Maintenance'],
    datasets: [
      {
        data: data.departmentRatings || [85, 92, 78, 88],
        backgroundColor: [
          '#3b82f6',
          '#10b981',
          '#f59e0b',
          '#8b5cf6',
        ],
        borderColor: [
          '#2563eb',
          '#059669',
          '#d97706',
          '#7c3aed',
        ],
        borderWidth: 3,
      },
    ],
  };

  return (
    <div className="dashboard">
      <div className="dashboard-header">
        <h1>Hotel Feedback Dashboard</h1>
        <p>Real-time insights into guest satisfaction</p>
      </div>

      <div className="stats-grid">
        <div className="stat-card">
          <div className="stat-icon">⭐</div>
          <div className="stat-content">
            <h3>4.3</h3>
            <p>Average Rating</p>
          </div>
        </div>
        <div className="stat-card">
          <div className="stat-icon">📊</div>
          <div className="stat-content">
            <h3>1,247</h3>
            <p>Total Reviews</p>
          </div>
        </div>
        <div className="stat-card">
          <div className="stat-icon">😊</div>
          <div className="stat-content">
            <h3>89%</h3>
            <p>Satisfaction Rate</p>
          </div>
        </div>
        <div className="stat-card">
          <div className="stat-icon">📈</div>
          <div className="stat-content">
            <h3>+12%</h3>
            <p>This Month</p>
          </div>
        </div>
      </div>

      <div className="charts-container">
        <div className="chart-section">
          <h2>Rating Analysis</h2>
          <div className="chart-row">
            <div className="chart-card">
              <div className="chart-header">
                <h3>Rating Distribution</h3>
                <span className="chart-subtitle">Guest ratings breakdown</span>
              </div>
              <div className="chart-content">
                <Bar data={ratingData} options={chartOptions} />
              </div>
            </div>
            <div className="chart-card">
              <div className="chart-header">
                <h3>Monthly Trends</h3>
                <span className="chart-subtitle">Average rating over time</span>
              </div>
              <div className="chart-content">
                <Line data={monthlyData} options={chartOptions} />
              </div>
            </div>
          </div>
        </div>

        <div className="chart-section">
          <h2>Department Performance</h2>
          <div className="chart-row">
            <div className="chart-card large">
              <div className="chart-header">
                <h3>Department Satisfaction</h3>
                <span className="chart-subtitle">Performance by department</span>
              </div>
              <div className="chart-content">
                <Doughnut 
                  data={departmentData} 
                  options={{
                    ...chartOptions,
                    plugins: {
                      ...chartOptions.plugins,
                      legend: {
                        position: 'right',
                        labels: {
                          usePointStyle: true,
                          padding: 20,
                        },
                      },
                    },
                  }} 
                />
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Dashboard;
