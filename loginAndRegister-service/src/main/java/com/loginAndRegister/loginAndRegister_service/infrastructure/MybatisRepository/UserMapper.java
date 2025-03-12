package com.loginAndRegister.loginAndRegister_service.infrastructure.MybatisRepository;

import com.loginAndRegister.loginAndRegister_service.Domain.model.User;
import org.apache.ibatis.annotations.*;
import org.apache.ibatis.type.JdbcType;


@Mapper
public interface UserMapper {

    @Insert("INSERT INTO users(username, password, role) VALUES(#{username}, #{password}, #{role})")
    void save(User user);

    @Select("SELECT id, username, password, role FROM users WHERE username = #{username}")
    User findByUsername(String username);

}
