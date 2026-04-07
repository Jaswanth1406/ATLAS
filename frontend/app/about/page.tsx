export default function AboutPage() {
  return (
    <div className="about-page">
      <div className="page-header">
        <h1>
          About <span className="gradient-text">ATLAS</span>
        </h1>
        <p>
          Adaptive Thresholding with Language-Augmented Sensing — an intelligent
          road segmentation system.
        </p>
      </div>

      {/* Architecture */}
      <h2>🏗️ Pipeline Architecture</h2>
      <div className="architecture-flow">
        <div className="arch-step">
          📷 Input Image
        </div>
        <span className="arch-arrow">→</span>
        <div className="arch-step">
          🔧 Preprocessing
          <span className="step-time">Resize + Normalize</span>
        </div>
        <span className="arch-arrow">→</span>
        <div className="arch-step">
          🧠 UNet Model
          <span className="step-time">ResNet34 Encoder</span>
        </div>
        <span className="arch-arrow">→</span>
        <div className="arch-step">
          📊 Sigmoid + Threshold
          <span className="step-time">Binary Mask</span>
        </div>
        <span className="arch-arrow">→</span>
        <div className="arch-step">
          🎨 Overlay
          <span className="step-time">Visualization</span>
        </div>
      </div>

      {/* Method comparison */}
      <h2>🔬 Segmentation Methods</h2>
      <p>
        ATLAS explored five adaptive thresholding techniques alongside the deep
        learning model. The UNet model significantly outperforms all classical
        approaches.
      </p>
      <div className="methods-table-wrapper">
        <table className="methods-table">
          <thead>
            <tr>
              <th>Method</th>
              <th>Avg IoU</th>
              <th>Speed</th>
              <th>Best Use Case</th>
              <th>Type</th>
            </tr>
          </thead>
          <tbody>
            <tr className="highlight-row">
              <td>
                <strong>UNet (ResNet34)</strong> ⭐
              </td>
              <td>
                <strong>0.85+</strong>
              </td>
              <td>~15ms</td>
              <td>All conditions</td>
              <td>
                <span className="badge badge-purple">Deep Learning</span>
              </td>
            </tr>
            <tr>
              <td>Adaptive Gaussian</td>
              <td>0.768</td>
              <td>16ms</td>
              <td>Shadows</td>
              <td>
                <span className="badge badge-green">Classical</span>
              </td>
            </tr>
            <tr>
              <td>Otsu</td>
              <td>0.756</td>
              <td>12ms</td>
              <td>Bright scenes</td>
              <td>
                <span className="badge badge-green">Classical</span>
              </td>
            </tr>
            <tr>
              <td>Adaptive Mean</td>
              <td>0.742</td>
              <td>16ms</td>
              <td>General</td>
              <td>
                <span className="badge badge-green">Classical</span>
              </td>
            </tr>
            <tr>
              <td>Sauvola</td>
              <td>0.721</td>
              <td>45ms</td>
              <td>Night scenes</td>
              <td>
                <span className="badge badge-green">Classical</span>
              </td>
            </tr>
            <tr>
              <td>Niblack</td>
              <td>0.698</td>
              <td>43ms</td>
              <td>Edge emphasis</td>
              <td>
                <span className="badge badge-green">Classical</span>
              </td>
            </tr>
          </tbody>
        </table>
      </div>

      {/* Model Details */}
      <h2>📦 Model Details</h2>
      <div className="methods-table-wrapper">
        <table className="methods-table">
          <thead>
            <tr>
              <th>Property</th>
              <th>Value</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>Architecture</td>
              <td>UNet</td>
            </tr>
            <tr>
              <td>Encoder</td>
              <td>ResNet34 (ImageNet pre-trained)</td>
            </tr>
            <tr>
              <td>Input Size</td>
              <td>256 × 256 px</td>
            </tr>
            <tr>
              <td>Output</td>
              <td>Binary mask (road / non-road)</td>
            </tr>
            <tr>
              <td>Loss Function</td>
              <td>BCE + Dice (combined)</td>
            </tr>
            <tr>
              <td>Export Format</td>
              <td>TorchScript (.pt)</td>
            </tr>
            <tr>
              <td>Model Size</td>
              <td>~93 MB</td>
            </tr>
          </tbody>
        </table>
      </div>

      {/* Team */}
      <h2>👥 Team</h2>
      <div className="team-grid">
        <div className="team-card">
          <div className="team-avatar">JP</div>
          <h4>Jaswanth Prasanna V</h4>
          <p>Rajalakshmi Institute of Technology</p>
        </div>
        <div className="team-card">
          <div className="team-avatar">DR</div>
          <h4>Divya R</h4>
          <p>Rajalakshmi Institute of Technology</p>
        </div>
        <div className="team-card">
          <div className="team-avatar">HK</div>
          <h4>Haripriya K</h4>
          <p>Rajalakshmi Institute of Technology</p>
        </div>
      </div>
    </div>
  );
}
