-- 添加默认用户用于测试
-- 使用方法: mysql -u root -p agentic_rag < scripts/add_default_user.sql

USE agentic_rag;

-- 添加默认用户（ID=1）
INSERT INTO users (id, username, email, hashed_password, is_active, created_at)
VALUES (1, 'default_user', 'default@example.com', 'test_password', TRUE, NOW())
ON DUPLICATE KEY UPDATE
    username = 'default_user',
    email = 'default@example.com';

-- 添加默认用户画像
INSERT INTO user_profiles (user_id, preferred_language, interaction_style, created_at)
VALUES (1, 'zh-CN', 'detailed', NOW())
ON DUPLICATE KEY UPDATE
    preferred_language = 'zh-CN';

-- 查看添加的用户
SELECT * FROM users WHERE id = 1;
SELECT * FROM user_profiles WHERE user_id = 1;
