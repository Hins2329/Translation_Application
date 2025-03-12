package com.loginAndRegister.loginAndRegister_service.infrastructure.MybatisRepository;

import com.loginAndRegister.loginAndRegister_service.Domain.model.User;

public interface UserRepository {
    void save(User user);
    String check(String username);

    User findByUsername(String username);
}
