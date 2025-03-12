package com.loginAndRegister.loginAndRegister_service.Service;
import com.loginAndRegister.loginAndRegister_service.infrastructure.MybatisRepository.UserMapper;
import com.loginAndRegister.loginAndRegister_service.infrastructure.MybatisRepository.UserRepositoryImpl;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.security.crypto.bcrypt.BCryptPasswordEncoder;
import com.loginAndRegister.loginAndRegister_service.Domain.model.User;
import org.springframework.stereotype.Service;

@Service
public class UserService {

    private final UserMapper userMapper;
    private final UserRepositoryImpl userRepository;
    private final BCryptPasswordEncoder passwordEncoder;

    @Autowired
    public UserService(UserMapper userMapper, UserRepositoryImpl userRepository1, BCryptPasswordEncoder passwordEncoder) {
        this.userRepository = userRepository1;
        this.userMapper = userMapper;
        this.passwordEncoder = passwordEncoder;
    }

    public void registerUser(User user){
        String hashedPassword = passwordEncoder.encode(user.getPassword());
        user.setPassword(hashedPassword);
        userRepository.save(user);
    }

    public User findByUsername(String username) {
        System.out.println("UserService");
        return userRepository.findByUsername(username);
    }


    public boolean verifyPassword(String rawPassword, String hashedPassword) {
        return passwordEncoder.matches(rawPassword,hashedPassword);
    }
}
