// 前端应用主文件
const API_BASE_URL = 'http://localhost:8001';  // 与后端端口一致

let currentVideoId = null;

// 初始化
document.addEventListener('DOMContentLoaded', () => {
    initTabs();
    initUpload();
    initDetectionConfig();
    loadVideoRecords();
});

// 标签页切换
function initTabs() {
    const tabBtns = document.querySelectorAll('.tab-btn');
    const tabContents = document.querySelectorAll('.tab-content');

    tabBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            const targetTab = btn.getAttribute('data-tab');
            
            // 更新按钮状态
            tabBtns.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            
            // 更新内容显示
            tabContents.forEach(content => {
                content.classList.remove('active');
            });
            document.getElementById(`${targetTab}-tab`).classList.add('active');
            
            // 加载对应数据
            if (targetTab === 'records') {
                loadVideoRecords();
            } else if (targetTab === 'detections') {
                loadDetections();
            }
        });
    });
}

// 初始化上传功能
function initUpload() {
    const uploadArea = document.getElementById('upload-area');
    const videoInput = document.getElementById('video-input');

    uploadArea.addEventListener('click', () => {
        videoInput.click();
    });

    uploadArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadArea.style.background = '#f0f0f0';
    });

    uploadArea.addEventListener('dragleave', () => {
        uploadArea.style.background = '#f9f9f9';
    });

    uploadArea.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadArea.style.background = '#f9f9f9';
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            handleVideoUpload(files[0]);
        }
    });

    videoInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            handleVideoUpload(e.target.files[0]);
        }
    });
}

// 处理视频上传
async function handleVideoUpload(file) {
    const formData = new FormData();
    formData.append('file', file);

    const progressBar = document.getElementById('upload-progress');
    const progressFill = progressBar.querySelector('.progress-fill');
    const statusDiv = document.getElementById('upload-status');

    progressBar.style.display = 'block';
    statusDiv.innerHTML = '<div class="status-message status-info">正在上传视频...</div>';

    try {
        const response = await fetch(`${API_BASE_URL}/api/videos/upload`, {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            throw new Error(`上传失败: ${response.statusText}`);
        }

        const data = await response.json();
        currentVideoId = data.video_id;

        progressFill.style.width = '100%';
        statusDiv.innerHTML = `
            <div class="status-message status-success">
                ✅ 视频上传成功！<br>
                文件名: ${data.filename}<br>
                视频ID: ${data.video_id}
            </div>
        `;

        // 显示检测配置
        document.getElementById('detection-config').style.display = 'block';

    } catch (error) {
        progressBar.style.display = 'none';
        statusDiv.innerHTML = `
            <div class="status-message status-error">
                ❌ 上传失败: ${error.message}
            </div>
        `;
    }
}

// 初始化检测配置
function initDetectionConfig() {
    const confSlider = document.getElementById('conf-threshold');
    const nmsSlider = document.getElementById('nms-threshold');
    const confValue = document.getElementById('conf-value');
    const nmsValue = document.getElementById('nms-value');
    const startBtn = document.getElementById('start-detection');

    confSlider.addEventListener('input', (e) => {
        confValue.textContent = parseFloat(e.target.value).toFixed(2);
    });

    nmsSlider.addEventListener('input', (e) => {
        nmsValue.textContent = parseFloat(e.target.value).toFixed(2);
    });

    startBtn.addEventListener('click', async () => {
        if (!currentVideoId) {
            alert('请先上传视频');
            return;
        }

        const confThreshold = parseFloat(confSlider.value);
        const nmsThreshold = parseFloat(nmsSlider.value);
        const inputShape = parseInt(document.getElementById('input-shape').value);
        const statusDiv = document.getElementById('detection-status');

        statusDiv.innerHTML = '<div class="status-message status-info">正在处理视频，请稍候...</div>';
        startBtn.disabled = true;

        try {
            const response = await fetch(
                `${API_BASE_URL}/api/videos/${currentVideoId}/detect?` +
                `conf_threshold=${confThreshold}&` +
                `nms_threshold=${nmsThreshold}&` +
                `input_shape=${inputShape}`,
                {
                    method: 'POST'
                }
            );

            if (!response.ok) {
                throw new Error(`检测失败: ${response.statusText}`);
            }

            const data = await response.json();
            statusDiv.innerHTML = `
                <div class="status-message status-success">
                    ✅ 检测完成！<br>
                    总帧数: ${data.total_frames}<br>
                    检测到目标的帧数: ${data.detected_frames}<br>
                    总检测数: ${data.total_detections}<br>
                    处理时间: ${data.processing_time.toFixed(2)}秒<br>
                    <a href="${API_BASE_URL}/api/videos/${currentVideoId}/download?detection_id=${data.detection_id}" 
                       target="_blank" class="btn btn-primary" style="margin-top: 10px; display: inline-block;">
                        下载处理后的视频
                    </a>
                </div>
            `;

            // 刷新检测结果列表
            loadDetections();

        } catch (error) {
            statusDiv.innerHTML = `
                <div class="status-message status-error">
                    ❌ 检测失败: ${error.message}
                </div>
            `;
        } finally {
            startBtn.disabled = false;
        }
    });
}

// 加载视频记录
async function loadVideoRecords() {
    const recordsList = document.getElementById('records-list');
    recordsList.innerHTML = '<div class="loading">加载中</div>';

    try {
        const response = await fetch(`${API_BASE_URL}/api/videos/records`);
        if (!response.ok) {
            throw new Error('加载失败');
        }

        const records = await response.json();
        
        if (records.length === 0) {
            recordsList.innerHTML = '<p style="text-align: center; color: #666; padding: 40px;">暂无视频记录</p>';
            return;
        }

        recordsList.innerHTML = records.map(record => `
            <div class="record-item">
                <h3>${record.filename}</h3>
                <p><strong>视频ID:</strong> ${record.id}</p>
                <p><strong>上传时间:</strong> ${new Date(record.upload_time).toLocaleString('zh-CN')}</p>
                <div class="meta">
                    ${record.video_type ? `<span>类型: ${record.video_type.toUpperCase()}</span>` : ''}
                    ${record.duration ? `<span>时长: ${record.duration.toFixed(1)}秒</span>` : ''}
                    ${record.fps ? `<span>帧率: ${record.fps.toFixed(1)} FPS</span>` : ''}
                    ${record.width && record.height ? `<span>分辨率: ${record.width}x${record.height}</span>` : ''}
                </div>
                <div style="margin-top: 15px;">
                    <button class="btn btn-primary" onclick="viewDetections(${record.id})">
                        查看检测结果
                    </button>
                    <button class="btn btn-secondary" onclick="deleteVideo(${record.id})" style="margin-left: 10px;">
                        删除
                    </button>
                </div>
            </div>
        `).join('');

    } catch (error) {
        recordsList.innerHTML = `
            <div class="status-message status-error">
                ❌ 加载失败: ${error.message}
            </div>
        `;
    }
}

// 加载检测结果
async function loadDetections() {
    const detectionsList = document.getElementById('detections-list');
    detectionsList.innerHTML = '<div class="loading">加载中</div>';

    try {
        // 先获取所有视频记录，然后获取每个视频的检测结果
        const videosResponse = await fetch(`${API_BASE_URL}/api/videos/records`);
        if (!videosResponse.ok) {
            throw new Error('加载失败');
        }

        const videos = await videosResponse.json();
        let allDetections = [];

        for (const video of videos) {
            const detectionsResponse = await fetch(`${API_BASE_URL}/api/videos/${video.id}/detections`);
            if (detectionsResponse.ok) {
                const detections = await detectionsResponse.json();
                allDetections = allDetections.concat(detections.map(d => ({ ...d, video_filename: video.filename })));
            }
        }

        // 按时间排序
        allDetections.sort((a, b) => new Date(b.created_at) - new Date(a.created_at));

        if (allDetections.length === 0) {
            detectionsList.innerHTML = '<p style="text-align: center; color: #666; padding: 40px;">暂无检测结果</p>';
            return;
        }

        detectionsList.innerHTML = allDetections.map(detection => {
            const summary = detection.detection_summary || {};
            const classCounts = summary.class_counts || {};
            
            return `
                <div class="detection-item">
                    <h3>检测结果 #${detection.id}</h3>
                    <p><strong>视频:</strong> ${detection.video_filename || `ID: ${detection.video_id}`}</p>
                    <p><strong>检测时间:</strong> ${new Date(detection.created_at).toLocaleString('zh-CN')}</p>
                    <div class="detection-stats">
                        <div class="stat-item">
                            <div class="stat-value">${detection.total_frames || 0}</div>
                            <div class="stat-label">总帧数</div>
                        </div>
                        <div class="stat-item">
                            <div class="stat-value">${detection.detected_frames || 0}</div>
                            <div class="stat-label">检测帧数</div>
                        </div>
                        <div class="stat-item">
                            <div class="stat-value">${detection.total_detections || 0}</div>
                            <div class="stat-label">总检测数</div>
                        </div>
                        <div class="stat-item">
                            <div class="stat-value">${detection.processing_time ? detection.processing_time.toFixed(2) : 0}</div>
                            <div class="stat-label">处理时间(秒)</div>
                        </div>
                    </div>
                    ${Object.keys(classCounts).length > 0 ? `
                        <div style="margin-top: 15px;">
                            <strong>类别统计:</strong>
                            <div style="margin-top: 10px;">
                                ${Object.entries(classCounts).map(([cls, count]) => 
                                    `<span style="background: #e9ecef; padding: 5px 12px; border-radius: 15px; margin-right: 10px; margin-bottom: 5px; display: inline-block;">
                                        ${cls}: ${count}
                                    </span>`
                                ).join('')}
                            </div>
                        </div>
                    ` : ''}
                    <div style="margin-top: 15px;">
                        <a href="${API_BASE_URL}/api/videos/${detection.video_id}/download?detection_id=${detection.id}" 
                           target="_blank" class="btn btn-primary">
                            下载处理后的视频
                        </a>
                    </div>
                </div>
            `;
        }).join('');

    } catch (error) {
        detectionsList.innerHTML = `
            <div class="status-message status-error">
                ❌ 加载失败: ${error.message}
            </div>
        `;
    }
}

// 查看检测结果
async function viewDetections(videoId) {
    // 切换到检测结果标签页
    document.querySelector('[data-tab="detections"]').click();
    
    // 可以在这里添加筛选功能，只显示该视频的检测结果
    setTimeout(() => {
        loadDetections();
    }, 100);
}

// 删除视频
async function deleteVideo(videoId) {
    if (!confirm('确定要删除这个视频记录及其所有检测结果吗？')) {
        return;
    }

    try {
        const response = await fetch(`${API_BASE_URL}/api/videos/${videoId}`, {
            method: 'DELETE'
        });

        if (!response.ok) {
            throw new Error('删除失败');
        }

        alert('删除成功');
        loadVideoRecords();
    } catch (error) {
        alert(`删除失败: ${error.message}`);
    }
}

// 刷新记录按钮
document.getElementById('refresh-records').addEventListener('click', () => {
    loadVideoRecords();
});
