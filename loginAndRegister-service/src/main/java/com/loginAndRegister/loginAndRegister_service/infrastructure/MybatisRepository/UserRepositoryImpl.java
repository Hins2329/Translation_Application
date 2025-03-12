package com.loginAndRegister.loginAndRegister_service.infrastructure.MybatisRepository;

import com.loginAndRegister.loginAndRegister_service.Domain.model.User;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Repository;

@Repository
public class UserRepositoryImpl implements UserRepository{

    private final UserMapper userMapper;


    @Autowired
    public UserRepositoryImpl(UserMapper userMapper) {
        this.userMapper = userMapper;
    }


    @Override
    public void save(User user) {
        userMapper.save(user);
    }


    @Override
    public User findByUsername(String username) {
        System.out.println("UserRepositoryImpl");
        return userMapper.findByUsername(username);
    }

    @Override
    public String check(String username) {
        return null;
    }


}
