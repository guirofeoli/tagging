<?php
// ===== LIBERAÇÃO DE CORS MULTIDOMÍNIO =====
$allowed_origins = [
    'https://www.ton.com.br',
    // 'http://localhost:3000', // adicione mais domínios se necessário
    // 'https://seu-outro-site.com'
];
if (isset($_SERVER['HTTP_ORIGIN']) && in_array($_SERVER['HTTP_ORIGIN'], $allowed_origins)) {
    header('Access-Control-Allow-Origin: ' . $_SERVER['HTTP_ORIGIN']);
}
header('Access-Control-Allow-Methods: POST, OPTIONS');
header('Access-Control-Allow-Headers: Content-Type');

// ===== HEADER PARA DESABILITAR CACHE =====
header('Cache-Control: no-cache, must-revalidate');
header('Expires: 0');

// ===== HANDLER DE OPTIONS PARA CORS =====
if ($_SERVER['REQUEST_METHOD'] === 'OPTIONS') {
    http_response_code(200);
    exit();
}

// ===== TOKEN DE SEGURANÇA =====
$token = 'Rofeoli@gui1';
if (!isset($_POST['token']) || $_POST['token'] !== $token) {
    http_response_code(403);
    die('Not authorized');
}

// ===== UPLOAD =====
$arquivos_salvos = [];
$arquivos_falha = [];
foreach($_FILES as $file) {
    $target = __DIR__ . '/' . basename($file['name']);
    if (move_uploaded_file($file['tmp_name'], $target)) {
        // Permissão para leitura universal (644)
        chmod($target, 0644);
        $arquivos_salvos[] = basename($file['name']);
    } else {
        $arquivos_falha[] = basename($file['name']);
    }
}

if (!empty($arquivos_falha)) {
    http_response_code(500);
    die('Erro ao salvar: ' . implode(', ', $arquivos_falha));
}

echo "OK\nArquivos salvos: " . implode(', ', $arquivos_salvos);
