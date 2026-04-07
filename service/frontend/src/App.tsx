import React, { useState, useEffect } from 'react';
import { Activity, TrendingUp, AlertTriangle, CheckCircle, Clock, BarChart3 } from 'lucide-react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, BarChart, Bar } from 'recharts';
import axios from 'axios';

const API_BASE_URL = 'http://localhost:8000/api/v1';

interface ScanResult {
  ticker: string;
  name: string;
  price: number | null;
  verdict: string;
  quality_score: number;
  tc_date: string | null;
  tc_uncertainty: { lower: number; upper: number; width: number } | null;
  hmm_regime: string | null;
  r_squared: number;
  data_points: number;
  scan_date: string;
}

interface Signal {
  ticker: string;
  price: number;
  quality: number;
  tc: string;
}

function App() {
  const [activeTab, setActiveTab] = useState<'signals' | 'scan' | 'history' | 'benchmark'>('signals');
  const [latestSignals, setLatestSignals] = useState<ScanResult[]>([]);
  const [loading, setLoading] = useState(true);
  const [tickers, setTickers] = useState('NVDA BTC-USD SPY');
  const [scanResults, setScanResults] = useState<ScanResult[]>([]);
  const [scanning, setScanning] = useState(false);

  // Load latest signals on mount
  useEffect(() => {
    fetchLatestSignals();
  }, []);

  const fetchLatestSignals = async () => {
    try {
      const response = await axios.get(`${API_BASE_URL}/signals`);
      setLatestSignals(response.data.signals || []);
    } catch (error) {
      console.error('Failed to fetch signals:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleScan = async () => {
    setScanning(true);
    try {
      const tickerList = tickers.split(/\s+/).filter(t => t.trim());
      const response = await axios.post(`${API_BASE_URL}/scan`, {
        tickers: tickerList,
        window_months: 12,
        domain: 'finance'
      });
      setScanResults(response.data.results);
    } catch (error) {
      console.error('Scan failed:', error);
      alert('Scan failed. Check console for details.');
    } finally {
      setScanning(false);
    }
  };

  const getVerdictIcon = (verdict: string) => {
    switch (verdict) {
      case 'BUBBLE':
        return <AlertTriangle className="text-red-500" />;
      case 'POSSIBLE':
        return <Activity className="text-yellow-500" />;
      default:
        return <CheckCircle className="text-green-500" />;
    }
  };

  const getVerdictColor = (verdict: string) => {
    switch (verdict) {
      case 'BUBBLE':
        return 'bg-red-100 text-red-800';
      case 'POSSIBLE':
        return 'bg-yellow-100 text-yellow-800';
      default:
        return 'bg-green-100 text-green-800';
    }
  };

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <header className="bg-white shadow">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <TrendingUp className="h-8 w-8 text-blue-600" />
              <div>
                <h1 className="text-2xl font-bold text-gray-900">PhaseBreak Dashboard</h1>
                <p className="text-sm text-gray-500">LPPLS Phase Transition Detection</p>
              </div>
            </div>
            <div className="flex items-center space-x-2 text-sm text-gray-600">
              <Clock className="h-4 w-4" />
              <span>v2.0.0</span>
            </div>
          </div>
        </div>
      </header>

      {/* Navigation Tabs */}
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 mt-4">
        <nav className="flex space-x-4">
          {[
            { id: 'signals' as const, label: 'Latest Signals', icon: <Activity className="h-4 w-4" /> },
            { id: 'scan' as const, label: 'Scan Assets', icon: <BarChart3 className="h-4 w-4" /> },
            { id: 'history' as const, label: 'History', icon: <Clock className="h-4 w-4" /> },
            { id: 'benchmark' as const, label: 'Benchmark', icon: <CheckCircle className="h-4 w-4" /> },
          ].map(tab => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              className={`flex items-center space-x-2 px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
                activeTab === tab.id
                  ? 'bg-blue-600 text-white'
                  : 'bg-white text-gray-700 hover:bg-gray-100'
              }`}
            >
              {tab.icon}
              <span>{tab.label}</span>
            </button>
          ))}
        </nav>
      </div>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
        {loading ? (
          <div className="text-center py-12">
            <div className="inline-block animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600"></div>
            <p className="mt-4 text-gray-600">Loading signals...</p>
          </div>
        ) : (
          <>
            {/* Signals Tab */}
            {activeTab === 'signals' && (
              <div className="space-y-6">
                {/* Summary Cards */}
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                  <div className="bg-white rounded-lg shadow p-6">
                    <div className="flex items-center justify-between">
                      <div>
                        <p className="text-sm text-gray-600">Bubble Signals</p>
                        <p className="text-3xl font-bold text-red-600">
                          {latestSignals.filter(s => s.verdict === 'BUBBLE').length}
                        </p>
                      </div>
                      <AlertTriangle className="h-12 w-12 text-red-500" />
                    </div>
                  </div>
                  <div className="bg-white rounded-lg shadow p-6">
                    <div className="flex items-center justify-between">
                      <div>
                        <p className="text-sm text-gray-600">Possible Signals</p>
                        <p className="text-3xl font-bold text-yellow-600">
                          {latestSignals.filter(s => s.verdict === 'POSSIBLE').length}
                        </p>
                      </div>
                      <Activity className="h-12 w-12 text-yellow-500" />
                    </div>
                  </div>
                  <div className="bg-white rounded-lg shadow p-6">
                    <div className="flex items-center justify-between">
                      <div>
                        <p className="text-sm text-gray-600">No Bubble</p>
                        <p className="text-3xl font-bold text-green-600">
                          {latestSignals.filter(s => s.verdict === 'NO_BUBBLE').length}
                        </p>
                      </div>
                      <CheckCircle className="h-12 w-12 text-green-500" />
                    </div>
                  </div>
                </div>

                {/* Signals Table */}
                <div className="bg-white rounded-lg shadow overflow-hidden">
                  <div className="px-6 py-4 border-b border-gray-200">
                    <h2 className="text-lg font-semibold text-gray-900">Latest Scan Results</h2>
                  </div>
                  <div className="overflow-x-auto">
                    <table className="min-w-full divide-y divide-gray-200">
                      <thead className="bg-gray-50">
                        <tr>
                          <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Ticker</th>
                          <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Verdict</th>
                          <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Quality</th>
                          <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">R²</th>
                          <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">tc Date</th>
                          <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">HMM Regime</th>
                          <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Price</th>
                        </tr>
                      </thead>
                      <tbody className="bg-white divide-y divide-gray-200">
                        {latestSignals.map((signal, idx) => (
                          <tr key={idx} className="hover:bg-gray-50">
                            <td className="px-6 py-4 whitespace-nowrap font-medium text-gray-900">{signal.ticker}</td>
                            <td className="px-6 py-4 whitespace-nowrap">
                              <div className="flex items-center space-x-2">
                                {getVerdictIcon(signal.verdict)}
                                <span className={`px-2 py-1 text-xs font-medium rounded ${getVerdictColor(signal.verdict)}`}>
                                  {signal.verdict}
                                </span>
                              </div>
                            </td>
                            <td className="px-6 py-4 whitespace-nowrap">
                              <div className="flex items-center">
                                <div className="w-16 bg-gray-200 rounded-full h-2 mr-2">
                                  <div
                                    className="bg-blue-600 h-2 rounded-full"
                                    style={{ width: `${Math.min(100, signal.quality_score * 100)}%` }}
                                  ></div>
                                </div>
                                <span className="text-sm text-gray-700">{signal.quality_score.toFixed(3)}</span>
                              </div>
                            </td>
                            <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-700">{signal.r_squared.toFixed(3)}</td>
                            <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-700">
                              {signal.tc_date || '—'}
                            </td>
                            <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-700">
                              {signal.hmm_regime || '—'}
                            </td>
                            <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-700">
                              {signal.price ? `$${signal.price.toFixed(2)}` : '—'}
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              </div>
            )}

            {/* Scan Tab */}
            {activeTab === 'scan' && (
              <div className="space-y-6">
                <div className="bg-white rounded-lg shadow p-6">
                  <h2 className="text-lg font-semibold text-gray-900 mb-4">Scan Assets</h2>
                  <div className="flex space-x-4">
                    <input
                      type="text"
                      value={tickers}
                      onChange={(e) => setTickers(e.target.value)}
                      placeholder="Enter tickers (space-separated)"
                      className="flex-1 px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                    />
                    <button
                      onClick={handleScan}
                      disabled={scanning}
                      className="px-6 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
                    >
                      {scanning ? 'Scanning...' : 'Scan'}
                    </button>
                  </div>
                  <p className="mt-2 text-sm text-gray-600">
                    Example: NVDA BTC-USD SPY TSLA
                  </p>
                </div>

                {scanResults.length > 0 && (
                  <div className="bg-white rounded-lg shadow overflow-hidden">
                    <div className="px-6 py-4 border-b border-gray-200">
                      <h2 className="text-lg font-semibold text-gray-900">Scan Results</h2>
                    </div>
                    <div className="p-6 space-y-4">
                      {scanResults.map((result, idx) => (
                        <div key={idx} className="border border-gray-200 rounded-lg p-4">
                          <div className="flex items-center justify-between mb-3">
                            <div className="flex items-center space-x-3">
                              <h3 className="text-lg font-bold text-gray-900">{result.ticker}</h3>
                              <span className={`px-2 py-1 text-xs font-medium rounded ${getVerdictColor(result.verdict)}`}>
                                {result.verdict}
                              </span>
                            </div>
                            {result.price && (
                              <span className="text-lg font-semibold text-gray-700">
                                ${result.price.toFixed(2)}
                              </span>
                            )}
                          </div>
                          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
                            <div>
                              <p className="text-gray-600">Quality Score</p>
                              <p className="font-semibold text-gray-900">{result.quality_score.toFixed(3)}</p>
                            </div>
                            <div>
                              <p className="text-gray-600">R²</p>
                              <p className="font-semibold text-gray-900">{result.r_squared.toFixed(3)}</p>
                            </div>
                            <div>
                              <p className="text-gray-600">tc Date</p>
                              <p className="font-semibold text-gray-900">{result.tc_date || '—'}</p>
                            </div>
                            <div>
                              <p className="text-gray-600">HMM Regime</p>
                              <p className="font-semibold text-gray-900">{result.hmm_regime || '—'}</p>
                            </div>
                          </div>
                          {result.tc_uncertainty && (
                            <div className="mt-3 text-sm text-gray-600">
                              tc uncertainty: [{result.tc_uncertainty.lower.toFixed(0)}, {result.tc_uncertainty.upper.toFixed(0)}] 
                              (width: {result.tc_uncertainty.width.toFixed(0)} days)
                            </div>
                          )}
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            )}

            {/* History Tab */}
            {activeTab === 'history' && (
              <div className="bg-white rounded-lg shadow p-6">
                <h2 className="text-lg font-semibold text-gray-900 mb-4">Signal History</h2>
                <p className="text-gray-600">Run the monitor a few times to populate history.</p>
              </div>
            )}

            {/* Benchmark Tab */}
            {activeTab === 'benchmark' && (
              <div className="space-y-6">
                <div className="bg-white rounded-lg shadow p-6">
                  <h2 className="text-lg font-semibold text-gray-900 mb-4">Benchmark Results</h2>
                  <p className="text-gray-600">58 episodes across 6 domains</p>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  <div className="bg-white rounded-lg shadow p-6">
                    <h3 className="text-sm font-semibold text-gray-900 mb-4">Precision & Recall</h3>
                    <ResponsiveContainer width="100%" height={200}>
                      <BarChart data={[
                        { metric: 'Precision', value: 0.76 },
                        { metric: 'Recall', value: 0.61 },
                        { metric: 'F1', value: 0.68 },
                      ]}>
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis dataKey="metric" />
                        <YAxis domain={[0, 1]} />
                        <Tooltip />
                        <Bar dataKey="value" fill="#3b82f6" />
                      </BarChart>
                    </ResponsiveContainer>
                  </div>

                  <div className="bg-white rounded-lg shadow p-6">
                    <h3 className="text-sm font-semibold text-gray-900 mb-4">Domain Performance</h3>
                    <ResponsiveContainer width="100%" height={200}>
                      <BarChart data={[
                        { domain: 'Finance', precision: 0.78, recall: 0.64 },
                        { domain: 'Commodities', precision: 0.67, recall: 0.67 },
                        { domain: 'Housing', precision: 0.67, recall: 0.33 },
                        { domain: 'Forward', precision: 1.0, recall: 1.0 },
                      ]}>
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis dataKey="domain" />
                        <YAxis domain={[0, 1]} />
                        <Tooltip />
                        <Legend />
                        <Bar dataKey="precision" fill="#3b82f6" name="Precision" />
                        <Bar dataKey="recall" fill="#10b981" name="Recall" />
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                </div>
              </div>
            )}
          </>
        )}
      </main>
    </div>
  );
}

export default App;
